import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from pauli_channel_approximation import PCA
from time import time
import scipy

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import scipy as sp
import numpy.linalg as lin
import scipy.linalg as slin

import os; os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
import pygsti

import pygsti
import numpy as np
import itertools
from functools import reduce

multi_kron = lambda *a: reduce(np.kron, a)

def change_basis(process, from_basis, to_basis):
    """
    PyGSTi 'std' basis is *row stacked* rather than column stacked, so need to defined a 'col' basis.
    """
    allowed_bases = ['col', 'std', 'gm', 'pp', 'qsim']
    if not ((from_basis in allowed_bases) or (to_basis in allowed_bases)):
        raise ValueError("Allowed bases are 'col', 'std', 'gm', 'pp', 'qsim' ")

    n_qubits = int(np.log(len(process)) / np.log(4))
    if from_basis == 'col' or to_basis == 'col':

        if len(process) != 4**n_qubits:
            raise ValueError('The dimension of the process matrix must be a power of 4.')
        def matrix_unit(n,i,j):
            unit = np.zeros([n,n], dtype='complex')
            unit[i,j] = 1.
            return unit
        col_stacked_matrix_units = [matrix_unit(2**n_qubits, i, j) for j in range(2**n_qubits) for i in range(2**n_qubits)]

        from pygsti.objects import ExplicitBasis
        col = ExplicitBasis(col_stacked_matrix_units, ["myElement%d" % i for i in range(4**n_qubits)],
                                     name='col', longname='Column=-Stacked')
        if from_basis == 'col':
            from_basis = col
        if to_basis == 'col':
            to_basis = col

    if from_basis == 'qsim' or to_basis == 'qsim':
        n_qubits = int(np.log(len(process))/np.log(4))
        if len(process) != 4**n_qubits:
            raise ValueError('The dimension of the process matrix must be a power of 4.')
        sig0q = np.array([[1., 0], [0, 0]], dtype='complex')
        sigXq = np.array([[0, 1], [1, 0]], dtype='complex')/np.sqrt(2)
        sigYq = np.array([[0, -1], [1, 0]], dtype='complex') * 1.j / np.sqrt(2.)
        sig1q = np.array([[0, 0], [0, 1]], dtype='complex')
        qbasis = itertools.product([sig0q, sigXq, sigYq, sig1q], repeat=n_qubits)
        qbasis = [multi_kron(*x) for x in qbasis]

        from pygsti.objects import ExplicitBasis
        qsim = ExplicitBasis(qbasis, ["myElement%d" % i for i in range(4**n_qubits)],
                                     name='qsim', longname='Quantumsim')
        if from_basis == 'qsim':
            from_basis = qsim
        if to_basis == 'qsim':
            to_basis = qsim

    return pygsti.tools.basistools.change_basis(process, from_basis, to_basis)


def get_unitary(data, c_ind, epsilon1, delta1):
    u = np.eye(2, dtype='complex')
    a = epsilon1 * data.ambient_hamiltonian[0]
    h0 = a
    for t_ind in range(25):
        f = (1 + delta1) * data.controlset[c_ind][t_ind][0] * data.control_hamiltonians[0]
        g = (1 + delta1) * data.controlset[c_ind][t_ind][1] * data.control_hamiltonians[1]
        ham = h0 + f + g
        u = np.dot(slin.expm(-1.j * ham * data.dt), u)
    return u


sqrt_x = scipy.linalg.sqrtm(np.array([[0, 1], [1, 0]]))
psqrt_x = np.kron(sqrt_x.conj(), sqrt_x)
psqrt_x = change_basis(psqrt_x, 'col', 'pp')


if rank == 0:
    import dill
    with open('0_pickled_controls5.pkl', 'rb') as f:
        data0 = dill.load(f)
    with open('1_pickled_controls5.pkl', 'rb') as f:
        data1 = dill.load(f)

    with open('./figures/0_delta_1q.dat', 'rb') as f:
        ep0 = dill.load(f)
    with open('./figures/1_delta_1q.dat', 'rb') as f:
        ep1 = dill.load(f)
else:
    data0 = None
    data1 = None
    ep0 = None
    ep1 = None


def split(n, a):
    k, m = divmod(len(a), n)
    return np.array(list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


data0 = comm.bcast(data0, root=0)
data1 = comm.bcast(data1, root=0)
ep0 = comm.bcast(ep0, root=0)
ep1 = comm.bcast(ep1, root=0)


my_points = split(size, np.arange(len(data0.controlset)))[rank]
my_values = []

t0 = time()
num_points = len(my_points)
for p_ind, point in enumerate(my_points):
    t1 = time()
    ed = []
    for val in ep0['vals']:
        u = get_unitary(data1, point, 0, val)
        pu = change_basis(np.kron(u.conj(), u), 'col', 'pp')
        ed += [pygsti.tools.optools.diamonddist(psqrt_x, pu, 'pp')/2.]
        # ed += [0]
    my_values += [ed]
    delta_t = np.round(time() - t1)
    delta_T = np.round(time() - t0)
    print(f"Node {rank} finished {p_ind+1} of {num_points} in {delta_T} seconds. Remaining: {delta_T*(num_points-p_ind-1)/(p_ind+1)}. Min dnorm: {np.min(ed)}.")

all_values = comm.gather(my_values, root = 0)


if rank == 0:
    print(np.array(all_values).shape)
    all_values = [inner for outer in all_values for inner in outer]
    data = {ind: all_values[ind] for ind in range(len(data0.controlset))}
    data['epsilons'] = ep0['vals']
    with open('1q_dnorms_delta.dat', 'wb') as f:
        dill.dump(data, f)












