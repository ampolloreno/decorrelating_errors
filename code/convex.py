from scipy.misc import derivative
from itertools import product
from functools import reduce
from copy import deepcopy
from GRAPE import control_unitaries
from scipy.linalg import logm
import cvxpy as cp
import numpy as np

DEFAULT_TOL = 1e-6


def error(combo, controls, target_operator, control_hamiltonians, ambient_hamiltonian0, dt):
    """Return the Hamiltonian that when exponentiated gives the error unitary from the target."""
    adjoint_target = np.conj(target_operator.T)
    newcontrols = deepcopy(controls)
    ambient_hamiltonian = [deepcopy(ah).astype("float") for ah in ambient_hamiltonian0]
    combo = list(combo)
    if len(combo) == 2:
        combo = combo + combo[-1:]
    else:
        assert len(combo) == 5
        combo = combo[:2] + combo[2:3] * 2 + combo[3:4] * 2 + combo[4:]
        assert len(combo) == 7
    assert len(combo) == len(control_hamiltonians) + len(ambient_hamiltonian0)
    for cnum, value in enumerate(combo):
        cnum -= len(ambient_hamiltonian0)
        if cnum >= 0:
            newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
        if cnum < 0:
            idx = len(ambient_hamiltonian) - abs(cnum)
            ambient_hamiltonian[idx] *= float(value)
    # just check the first one
    assert np.isclose(ambient_hamiltonian[0], np.conj(ambient_hamiltonian[0].T)).all()
    for h in control_hamiltonians:
        assert np.isclose(h, np.conj(h.T)).all()
    step_unitaries = control_unitaries(ambient_hamiltonian,
                                       control_hamiltonians,
                                       newcontrols,
                                       dt)
    for u in step_unitaries:
        assert np.isclose(np.conj(u.T).dot(u), np.eye(int(np.sqrt(target_operator.size)))).all(), np.conj(u.T).dot(u)
    unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
    error_gen = logm(adjoint_target.dot(unitary))
    num_states = error_gen.shape[0]
    eye_proj = np.trace(np.dot(np.eye(num_states), error_gen))/num_states
    # entry = (error_gen[0, 0] - error_gen[1, 1])/2
    # error_gen[0, 0] = entry
    # error_gen[1, 1] = -entry
    error_gen -= eye_proj * np.eye(error_gen.shape[0])
    return -1.j * error_gen


def deg_deriv(controlset, target, control_hamiltonians, ambient_hamiltonian0, dt, deg):
    ds = []
    if target.shape == (2, 2):
        point = np.array([0, 0])
    else:
        point = np.array([0, 0, 0, 0, 0])
    for control in controlset:
        d = compute_ith_derivative(lambda x: error(x, control, target, control_hamiltonians, ambient_hamiltonian0, dt), point, tuple(), deg, target.size)
        ds.append(d)
    ds = np.array(ds)
    return ds


def all_derivs(controlset, target, control_hamiltonians, ambient_hamiltonian0, dt, i):
    res = []
    for deg in range(i+1):
        ds = deg_deriv(controlset, target, control_hamiltonians, ambient_hamiltonian0, dt, deg)
        res.append(ds)
    return res


def partial(func, point, index, args):
    def f(x):
        return func([p if i != index else x for i, p in enumerate(point)])
    return derivative(f, point[index], n=1, args=args,  dx=1.0/2**16)


def compute_partial(f, point, tup, args):
    """Compute the derivative of f at point of order tup, order must be positive. Tup tells you which derivatives to take
    i.e. 0,0,1 says take the derivative of the first element, then the first, then the second.
    args should be any accessory data to pass along."""
    if len(tup) == 1:
        return partial(f, point, tup[0], args)
    # I think this assumes everything is at 0
    return compute_partial(lambda x: partial(f, x, tup[0], args), point, tup[1:], args)


def compute_ith_derivative(f, point, args, i, matsize):
    if i == 0:
        return np.array(f(point, *args)).flatten().reshape(1, -1)
    indices = list(range(len(point)))
    tups = product(*[indices]*i)
    res = np.zeros(tuple([len(point)]*i+[matsize]), dtype='complex')
    for tup in tups:
        res[tup] = compute_partial(f, point, tup, args).flatten()
    return res


def optimal_weights_1st_order(derivs, l, tol=DEFAULT_TOL):
    mini = float('inf')
    res = None
    for i in range(len(derivs[0])):
        if i % 100:
            print(f"Done with convex problem {i} out of {len(derivs[0])}")
        ham_consts = []
        for deriv in derivs:
            ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
        omega = cp.Variable(len(derivs[0]))
        t = cp.Variable(1)
        constraints = [0 <= omega, omega <= 1, sum(omega) == 1, t >= 0]
        constraints += [omega[i] >= l * cp.inv_pos(t)]
        equalities = ham_consts[:-1]
        for ham_const in equalities:
            constraints += [np.real(ham_const) * omega == 0]
            constraints += [np.imag(ham_const) * omega == 0]
        first_order = ham_consts[-1]
        #first_order = compute_first_order_term(derivs)
        objective = cp.Minimize(cp.norm(np.real(first_order) * omega) + cp.norm(np.imag(first_order) * omega) + t)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.CVXOPT, kktsolver=cp.ROBUST_KKTSOLVER, abstol=tol, abstol_inacc=tol)
        if result < mini and omega.value is not None:
            mini = result
            res = omega.value
    return res


def optimal_weights_no_constraints(derivs, l, tol=DEFAULT_TOL):
    print("Starting optimal weights no constraints")
    mini = float('inf')
    res = None
    ham_consts = []
    for deriv in derivs:
        ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
    omega = cp.Variable(len(derivs[0]))
    constraints = [0 <= omega, omega <= 1, sum(omega)==1]
    equalities = ham_consts[:-1]
    for ham_const in equalities:
        constraints += [np.real(ham_const)*omega == 0]
        constraints += [np.imag(ham_const)*omega == 0]
    objective = cp.Minimize(cp.norm(np.real(ham_consts[-1])*omega) + cp.norm(np.imag(ham_consts[-1])*omega))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.CVXOPT, kktsolver=cp.ROBUST_KKTSOLVER, abstol=tol, abstol_inacc=tol)
    return omega.value


def optimal_weights_1st_order_no_constraints(derivs, l, tol=DEFAULT_TOL):
    print("Starting optimal weights no constraints 1st order")
    mini = float('inf')
    res = None
    ham_consts = []
    for deriv in derivs:
        ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
    omega = cp.Variable(len(derivs[0]))
    constraints = [0 <= omega, omega <= 1, sum(omega) == 1]
    equalities = ham_consts[:-1]
    # for ham_const in equalities:
    #     constraints += [np.real(ham_const) * omega == 0]
    #     constraints += [np.imag(ham_const) * omega == 0]
    first_order = ham_consts[-1]
    objective = cp.Minimize(cp.norm(np.real(first_order) * omega) + cp.norm(np.imag(first_order) * omega))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.CVXOPT, kktsolver=cp.ROBUST_KKTSOLVER, abstol=tol, abstol_inacc=tol)
    return omega.value


def compute_first_order_term(derivs):
    dim = int(np.sqrt(derivs[0][0].shape[1]))
    hdh = [np.matmul(derivs[0][i].reshape(dim, dim), d.reshape((dim, dim, -1))) for i, d in enumerate(derivs[1])]
    hdh = [[a[:, :, 0], a[:, :, 1]] for a in hdh]

    dh = [d.reshape((dim, dim, -1)) for d in derivs[1]]
    dh = [[a[:, :, 0], a[:, :, 1]] for a in dh]

    h = [d.reshape((dim, dim, -1)) for d in derivs[0]]

    dhh = [[np.kron(h[i].T, el[0]), np.kron(h[i].T, el[1])] for i, el in enumerate(dh)]
    hdh = [[np.kron(np.eye(dim), el[0]), np.kron(np.eye(dim), el[1])] for i, el in enumerate(hdh)]
    dh = [[np.kron(np.eye(dim), el[0]), np.kron(np.eye(dim), el[1])] for el in dh]

    dhh = np.matrix(np.array(dhh).T.reshape(-1, len(derivs[0])))
    hdh = np.matrix(np.array(hdh).T.reshape(-1, len(derivs[0])))
    dh = np.matrix(np.array(dh).T.reshape(-1, len(derivs[0])))

    first_order = hdh + dhh + dh
    return first_order


def optimal_weights(derivs, l, tol=DEFAULT_TOL):
    mini = float('inf')
    res = None
    for i in range(len(derivs[0])):
        if i % 100:
            print(f"Done with convex problem {i} out of {len(derivs[0])}")
        ham_consts = []
        for deriv in derivs:
            ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
        omega = cp.Variable(len(derivs[0]))
        t = cp.Variable(1)
        constraints = [0 <= omega, omega <= 1, sum(omega)==1, t>=0]
        constraints += [omega[i] >= l*cp.inv_pos(t)]
        equalities = ham_consts[:-1]
        for ham_const in equalities:
            constraints += [np.real(ham_const)*omega == 0]
            constraints += [np.imag(ham_const)*omega == 0]
        objective = cp.Minimize(cp.norm(np.real(ham_consts[-1])*omega) + cp.norm(np.imag(ham_consts[-1])*omega) + t)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.CVXOPT, kktsolver=cp.ROBUST_KKTSOLVER, abstol=tol, abstol_inacc=tol)
        if result < mini and omega.value is not None:
            mini = result
            res = omega.value
    return res