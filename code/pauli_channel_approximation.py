import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GRAPE import GRAPE, control_unitaries, adjoint
import numpy as np
import dill
from functools import reduce
from tqdm import tqdm
import time as timemod
from convex import all_derivs, optimal_weights, optimal_weights_1st_order
from mpi4py import MPI
import scipy
import os
from itertools import product

COMM = MPI.COMM_WORLD

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.j], [1.j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]


def generate_indices(num_points, order_desired):
    num_indices = len(order_desired)
    tuples = product(range(num_points), repeat=num_indices)
    indices = [sum([num_points**(num_indices - 1 - order_desired[i]) * t[i] for i in range(num_indices)]) for t in tuples]
    return indices


def error_unitary(unitary, target):
    """
    Constructs the superoperator for the error unitary corresponding propagating forward by target,
     and back by unitary.

    :param np.array unitary: Control unitary realized in the computational basis.
    :param np.array target: The desired unitary in the computational basis.
    :return: The superoperator corresponding to the unitary postrotational error.
    :rtype: numpy.array
    """
    error = adjoint(unitary).dot(target)
    return np.kron(np.conj(error), error)


def off_diagonal_projection(sop):
    """
    Computes a quantity proportional to the sum of the magnitudes of terms corresponding to non-stochastic evolution.

    :param numpy.ndarry sop: Representation our operator as a superoperator, e.g. operating on vectorized density
     matrices.
    """
    basis = PAULIS
    for _ in range(int(np.log2(sop.shape[0]) / 2.0) - 1):
        basis = [np.kron(base, pauli) for pauli in PAULIS for base in basis]
    off_diagonal = 0
    for i, base1 in enumerate(basis):
        for j, base2 in enumerate(basis):
            if i != j:
                off_diagonal += np.abs(np.trace(adjoint(np.kron(base1, base2)).dot(sop))) ** 2
    return off_diagonal


def error_on_controls(ambient_hamiltonian, control_hamiltonians, controls, dt,
                      target_operator):
    unitary = reduce(lambda a, b: a.dot(b),
                     control_unitaries(ambient_hamiltonian, control_hamiltonians,
                                       controls, dt))
    error = error_unitary(unitary, target_operator)
    return error


class PCA(object):
    """Class to perform Pauli Channel Approximations- i.e. pick out weights for families of controls that make them look
     most like Pauli Channels."""

    def __init__(self, num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
                 num_steps, time, threshold, detunings):
        self.seed = 138
        np.random.seed(self.seed)
        self.start = timemod.time()
        controlset = []
        dt = time / num_steps
        self.num_controls = num_controls
        for i in tqdm(range(num_controls)):
            print("CONTROL {}".format(i))
            random_detunings = []
            for detuning in detunings:
                random_detunings.append((detuning[0], detuning[1]))
            import sys
            sys.stdout.flush()
            result = GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator,
                           num_steps, time, threshold, random_detunings)
            controlset.append(result.reshape(-1, len(control_hamiltonians)))
        self.controlset = controlset
        self.detunings = detunings
        self.target_operator = target_operator
        self.dt = dt
        self.ambient_hamiltonian = ambient_hamiltonian
        self.control_hamiltonians = control_hamiltonians

    def assign_weights(self, l1=0, l2=1E-3):
        derivs = all_derivs(self.controlset, self.target_operator, self.control_hamiltonians, self.ambient_hamiltonian,
                            self.dt, 1)
        weights = optimal_weights_1st_order(derivs, l1)
        weights_0 = optimal_weights(derivs[:1], l2)
        self.derivs = derivs
        self.weights = weights
        self.weights_0 = weights_0
        print("Assigned weights.")


def compute_dpn_and_fid(data):
    from copy import deepcopy
    import sys
    controlset, ambient_hamiltonian0, combo, dt, control_hamiltonians, target_operator, probs = data
    print("DOING COMBO {}".format(combo))
    sys.stdout.flush()
    fidelities = []
    projs = []
    sops = []
    controlset_unitaries = []
    #
    #
    # for i, com in enumerate(combo):
    #     if i != 0 and com != 0:
    #         return 0
    for controls in controlset:
        newcontrols = deepcopy(controls)
        ambient_hamiltonian = [deepcopy(ah).astype("float") for ah in ambient_hamiltonian0]
        for cnum, value in enumerate(combo):
            cnum -= len(ambient_hamiltonian0)
            if cnum >= 0:
                newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
            if cnum < 0:
                ambient_hamiltonian[cnum] *= float(value)
        step_unitaries = control_unitaries(ambient_hamiltonian,
                                           control_hamiltonians, newcontrols,
                                           dt)
        unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
        sop = error_unitary(unitary, target_operator)
        sops.append(sop)
        projs.append(off_diagonal_projection(sop))
        controlset_unitaries.append(unitary)
        fidelity = np.trace(adjoint(target_operator).dot(unitary))/target_operator.shape[0]
        fidelity *= np.conj(fidelity)
        fidelities.append(fidelity)
    # print(sops)
    # print(probs)
    # print([prob * sops[i] for i, prob in enumerate(probs)])
    avg_sop = reduce(lambda a, b: a + b, [prob * sops[i] for i, prob in enumerate(probs)])
    balanced = reduce(lambda a, b: a + b,
                      [probs[i] * np.kron(np.conj(unitary), unitary) for i, unitary in
                       enumerate(controlset_unitaries)])
    balanced_fidelity = np.trace(
        adjoint(balanced).dot(np.kron(np.conj(target_operator), target_operator)))/(target_operator.shape[0])**2

    fidelities.append(balanced_fidelity)
    projs.append(off_diagonal_projection(avg_sop))

    fidelities = np.array(fidelities).T
    projs = np.array(projs).T

    return projs, fidelities


def gen_1q():
    COMM = MPI.COMM_WORLD
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ambient_hamiltonian = [Z]
    control_hamiltonians = [X, Y]
    detunings = [(1E-3, 1), (1E-3,  2)]
    target_operator = scipy.linalg.sqrtm(X)
    time = np.pi
    num_steps = 25
    threshold = 1 - .001
    num_controls = 10
    pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
              num_steps, time, threshold, detunings)

    if COMM.rank == 0:
        i = 0
        while os.path.exists("pickled_controls%s.pkl" % i):
            i += 1
        fh = open("just_pickled_controls%s.pkl" % i, "wb")
        dill.dump(pca, fh)
        pca.assign_weights()
        fh.close()
        fh = open("pickled_controls%s.pkl" % i, "wb")
        dill.dump(pca, fh)
        fh.close()


def gen_2q():
    COMM = MPI.COMM_WORLD
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    IZ = np.kron(I, Z)
    ZI = np.kron(Z, I)
    XI = np.kron(X, I)
    IX = np.kron(I, X)
    IY = np.kron(I, Y)
    YI = np.kron(Y, I)
    ZZ = np.kron(Z, Z)
    entangle_ZZ = np.array([[1, 0, 0, 0], [0, 1.j, 0, 0], [0, 0, 1.j, 0], [0, 0, 0, 1]])
    ambient_hamiltonian = [IZ, ZI]
    control_hamiltonians = [IX, IY, XI, YI, ZZ]
    detunings = [(.001, 1), (.001, 1), (.001, 2), (.001, 2), (.001, 1)]
    target_operator = entangle_ZZ
    time = 2. * np.pi
    num_steps = 40
    threshold = 1 - .001
    num_controls = 200
    pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
              num_steps, time, threshold, detunings)
    if COMM.rank == 0:
        i = 0
        while os.path.exists("pickled_controls%s.pkl" % i):
            i += 1
        fh = open("just_pickled_controls%s.pkl" % i, "wb")
        dill.dump(pca, fh)
        pca.assign_weights()
        fh.close()
        fh = open("pickled_controls%s.pkl" % i, "wb")
        dill.dump(pca, fh)
        fh.close()


if __name__ == '__main__':
    gen_2q()
