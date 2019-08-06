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

def print(*args, **kwargs):
    if rank == 0:
        return __builtin__.print(*args, **kwargs)

# Import the data
import dill
filenames = {0: '0_pickled_controls106.pkl', 1: '1_pickled_controls106.pkl'}
with open(filenames[1], 'rb') as f:
    data = dill.load(f)

# Divide list into n approximately equally sized chunks
def split(n, a):
    k, m = divmod(len(a), n)
    return np.array(list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


# Get the unitary associated with a given index
def get_unitary(c_ind, epsilon1, epsilon2, delta1, delta2):
    u = np.eye(4, dtype='complex')
    
    a = epsilon1 * data.ambient_hamiltonian[0]
    b = epsilon2 * data.ambient_hamiltonian[1]
    c = data.ambient_hamiltonian[2] 
    h0 = a+b+c
    
    for t_ind in range(500):
        d = (1+delta2) * data.controlset[c_ind][t_ind][0] * data.control_hamiltonians[0]
        e = (1+delta2) * data.controlset[c_ind][t_ind][1] * data.control_hamiltonians[1] 
        f = (1+delta1) * data.controlset[c_ind][t_ind][2] * data.control_hamiltonians[2] 
        g = (1+delta1) * data.controlset[c_ind][t_ind][3] * data.control_hamiltonians[3] 
        ham = h0 + d + e + f + g
        u = np.dot(slin.expm(-1.j*ham*data.dt), u)    
    return u    

# for testing:
my_inds = split(size, np.arange(len(data.controlset)))[rank]
# my_inds = np.array(split(size, np.arange(size)))[rank]

# Get the process matrix for a given setting of the error parameters
def get_process(ep1, ep2, delta1, delta2):
    my_us = [get_unitary(ind, ep1, ep2, delta1, delta2) for ind in my_inds]
    my_probs = [data.probs[ind] for ind in my_inds]
    my_processes = [prob * pygsti.tools.optools.unitary_to_pauligate(u) for u, prob in zip(my_us, my_probs)]
    all_processes = comm.gather(my_processes, root = 0)
    if rank == 0:
        return sum([y for x in all_processes for y in x],0)
    else: 
        return None

if __name__ == '__main__':
    
    diamond_distances = []
    vals = np.linspace(-.01,.01,31)
    target_process = pygsti.tools.optools.unitary_to_pauligate(data.target_operator)

    for val in vals:
        t1 = time()    
        process = get_process(0,0,val,val)
        if rank == 0:  
            error = pygsti.tools.optools.diamonddist(target_process, process, 'pp')/2
            print(time() - t1, val, error)
            diamond_distances += [error]

    if rank == 0:
        plt.plot(vals, diamond_distances)
        plt.xlabel('Delta')
        plt.savefig('./figures/delta_correlated.pdf')
