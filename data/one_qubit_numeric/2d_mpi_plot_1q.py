# Standard imports
import numpy as np
import scipy as sp
import numpy.linalg as lin
import scipy.linalg as slin

# Set the environment variables
import os

os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Initialize the parallel processing
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Additional imports
from time import time
import pygsti

if rank == 0:
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt


def print(*args, root=0, **kwargs):
    if rank == root:
        import builtins
        return builtins.print(*args, **kwargs)


# Import the data
import dill

filenames = {0: '0_pickled_controls5.pkl', 1: '1_pickled_controls5.pkl'}
with open(filenames[1], 'rb') as f:
    data = dill.load(f)


# Divide list into n approximately equally sized chunks
def split(n, a):
    k, m = divmod(len(a), n)
    return np.array(list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


# Get the unitary associated with a given index
def get_unitary(c_ind, epsilon1, delta1):
    u = np.eye(2, dtype='complex')
    a = epsilon1 * data.ambient_hamiltonian[0]
    h0 = a
    for t_ind in range(25):
        f = (1 + delta1) * data.controlset[c_ind][t_ind][0] * data.control_hamiltonians[0]
        g = (1 + delta1) * data.controlset[c_ind][t_ind][1] * data.control_hamiltonians[1]
        ham = h0 + f + g
        u = np.dot(slin.expm(-1.j * ham * data.dt), u)
    return u


# for testing:
my_inds = split(size, np.arange(len(data.controlset)))[rank]


# my_inds = np.array(split(size, np.arange(size)))[rank]


# Get the process matrix for a given setting of the error parameters
def get_process(ep1, delta1, return_average=True, return_diamond_distance=False):
    my_us = [get_unitary(ind, ep1, delta1) for ind in my_inds]
    my_probs = [data.probs[ind] for ind in my_inds]

    target_process = pygsti.tools.optools.unitary_to_pauligate(data.target_operator)

    if return_average:
        my_processes = [prob * pygsti.tools.optools.unitary_to_pauligate(u) for u, prob in zip(my_us, my_probs)]
    else:
        my_processes = [pygsti.tools.optools.unitary_to_pauligate(u) for u in my_us]
        if return_diamond_distance:
            my_processes = [pygsti.tools.optools.diamonddist(target_process, process, 'pp') / 2. for process in
                            my_processes]

    all_processes = comm.gather(my_processes, root=0)
    if rank == 0:
        if return_average:
            return sum([y for x in all_processes for y in x], 0)
        else:
            return np.array([y for x in all_processes for y in x])
    else:
        return None


def flatten(the_list):
    return [inner for outer in the_list for inner in outer]


if __name__ == '__main__':

    diamond_distances = []
    vals = np.linspace(-.002,.002,1001)
    target_process = pygsti.tools.optools.unitary_to_pauligate(data.target_operator)
    t0 = time()
    for v_ind, val in enumerate(vals):
        t1 = time()
        process = get_process(0, val)
        if rank == 0:
            error = pygsti.tools.optools.diamonddist(target_process, process, 'pp')/2
            t_total = np.round(time()-t0,3)
            t_remaining_estimate = np.round(t_total/(v_ind+1) * (len(vals) - (v_ind+1)),3)
            print(f"{v_ind+1}/{len(vals)} took {np.round(time() - t1,3)} seconds. Total: {t_total}  Remaining: {t_remaining_estimate}")
            print(f"     The diamond distance is {np.round(error, 5)} at epsilon = {np.round(val, 4)}.")
            diamond_distances += [error]

    if rank == 0:
        save_data = {'vals': vals, 'diamond_distances': diamond_distances}
        with open('./figures/1_delta_1q.dat', 'wb') as f:
            dill.dump(save_data, f)
        plt.plot(vals, diamond_distances)
        plt.xlabel('Delta')
        plt.ylabel('Diamond Distance')
        plt.savefig('./figures/1_epsilon_correlated_dense_1q.pdf')

    t1 = time()
    ddists = get_process(0,.01, False, True)
    print(ddists)
    if rank == 0:
        print(ddists)
        ddists = flatten(ddists)
        best_index = np.argmin(ddists)
        print(best_index, ddists[best_index])
        print(np.round(time() - t1, 1))

    target_process = pygsti.tools.optools.unitary_to_pauligate(data.target_operator)
    vals = np.linspace(-0.002, 0.002, 79)
    my_vals = split(size, vals)[rank]
    for index in [399, 39, 234, 252, 824]:
        my_ddists = []
        for val in my_vals:
            process = pygsti.tools.optools.unitary_to_pauligate(get_unitary(index,0, val))
            ddist = pygsti.tools.optools.diamonddist(process, target_process, 'pp')/2.
            my_ddists += [ddist]
        ddists = comm.gather(my_ddists, root = 0)

        if rank == 0:
            ddists = flatten(ddists)
            print(ddists)
            plt.plot(vals, ddists, label = str(index))
    if rank == 0:
        plt.legend()
        plt.savefig('./figures/several_1q.pdf')
















