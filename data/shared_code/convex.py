from scipy.misc import derivative
from itertools import product
from functools import reduce
from copy import deepcopy
from GRAPE import control_unitaries
from scipy.linalg import logm
import cvxpy as cp
import numpy as np
from scipy.linalg import norm

DEFAULT_TOL = 1e-8


def error(combo, controls, target_operator, control_hamiltonians, ambient_hamiltonian0, dt):
    """Return the Hamiltonian that when exponentiated gives the error unitary from the target."""
    adjoint_target = np.conj(target_operator.T)
    newcontrols = deepcopy(controls)
    ambient_hamiltonian = [deepcopy(ah).astype("complex") for ah in ambient_hamiltonian0]
    combo = list(combo)
    if len(combo) == 2:
        combo = combo + combo[-1:]
        assert len(combo) == 3
    else:
        assert len(combo) == 5
        combo = combo[:2] + combo[2:3] * 2 + combo[3:4] * 2 + combo[4:]
        assert len(combo) == 7
    assert len(combo) == len(control_hamiltonians) + len(ambient_hamiltonian0)
    assert len(combo) == len(control_hamiltonians) + len(ambient_hamiltonian0)
    for cnum, value in enumerate(combo):
        cnum -= len(ambient_hamiltonian0)
        if cnum >= 0:
            newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
        if cnum < 0:
            idx = len(ambient_hamiltonian) - abs(cnum)
            if cnum == -1 and len(combo) != 3: # not the one qubit case, in particular the uncontrolled 2q hamiltonian
                continue
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
    error_gen -= eye_proj * np.eye(error_gen.shape[0])
    return -1.j * error_gen


def deg_deriv(controlset, target, control_hamiltonians, ambient_hamiltonian0, dt, deg):
    ds = []
    if target.shape == (2, 2):
        point = np.array([0, 0])
    else:
        point = np.array([0, 0, 0, 0, 0])
    for i, control in enumerate(controlset):
        print("Control {} derivative.".format(i))
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


def optimal_weights(derivs, l2_constraint=False, l2_param=None, sparsity_param=None, sparsity=False, dt=None):
    """Super hacky, but this now takes the dt optionally, since we are moving from using the l2 norm to using
    the agi, and that requires us to exponentiate the error generator for some time."""
    mini = float('inf')
    res = None
    if sparsity:
        for i in range(len(derivs[0])):
            try:
                if i % 10 == 0:
                    print(f"Done with convex problem {i} out of {len(derivs[0])}")
                ham_consts = []
                for deriv in derivs:
                    ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
                omega = cp.Variable(len(derivs[0]))
                t = cp.Variable(1)
                constraints = [0 <= omega, omega <= 1, sum(omega) == 1, t >= 0]
                constraints += [omega[i] >= cp.inv_pos(t) * sparsity_param]
                equalities = ham_consts[:1]
                for ham_const in equalities:
                   constraints += [np.real(ham_const) * omega == 0]
                   constraints += [np.imag(ham_const) * omega == 0]

                objective_argument = (t
                                      + cp.norm(np.real(ham_consts[-1]) * omega)
                                      + cp.norm(np.imag(ham_consts[-1]) * omega))
                # TODO fix for sim
                if l2_constraint:
                    import scipy
                    from GRAPE import adjoint, control_unitaries
                    agis = []
                    for ham in derivs[0]:
                        unitary = scipy.linalg.expm(-1.j * np.reshape(ham, (4, 4)) * dt)
                        #only works for 1q rn
                        agis.append((2+np.trace(unitary)) / 6)
                    ham_norm = np.matrix(agis)
                    # What does this do if I don't write norm???
                    objective_argument += cp.norm(l2_param * ham_norm * omega)
                objective = cp.Minimize(objective_argument)

                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver=cp.MOSEK, verbose=True)
                if result < mini and omega.value is not None:
                    mini = result
                    res = omega.value
            except cp.SolverError:
                print("FAILED")
                continue
    else:
        ham_consts = []
        for deriv in derivs:
            ham_consts.append(np.matrix([d.flatten() for d in deriv]).T)
        omega = cp.Variable(len(derivs[0]))
        constraints = [0 <= omega, omega <= 1, sum(omega) == 1]
        equalities = ham_consts[:1]
        for ham_const in equalities:
            constraints += [np.real(ham_const) * omega == 0]
            constraints += [np.imag(ham_const) * omega == 0]

        objective_argument = (cp.norm(np.real(ham_consts[-1]) * omega)
                              + cp.norm(np.imag(ham_consts[-1]) * omega))
        if l2_constraint:
            import scipy
            from GRAPE import adjoint, control_unitaries
            agis = []
            for ham in derivs[0]:
                unitary = scipy.linalg.expm(-1.j * np.reshape(ham, (2, 2)) * dt)
                # only works for 1q rn
                agis.append((np.trace(np.kron(np.conj(unitary), unitary)) + 2)/6)
            ham_norm = np.matrix(agis)
            # What does this do if I don't write norm???
            objective_argument += cp.norm(l2_param * ham_norm * omega)
        objective = cp.Minimize(objective_argument)

        prob = cp.Problem(objective, constraints)
        _ = prob.solve(solver=cp.MOSEK, verbose=True)
        res = omega.value
    return res


