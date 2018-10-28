To generate two qubit controls, save them, and assign the optimal weights, it should currently be sufficient to do the following (assuming python 3.7)

```
pip install -r requirements.txt
python pauli_channel_approximation.py
```


This will need to run in a shell, so in order for it to run through an ssh connection, for instance, one will need to use screen or tmux. With screen one can do the following:

```
screen -S pca_generation
python pauli_channel_approximation.py
```

Following by Ctrl+D to detach. To view all current screens one can use:
```
screen -ls
```
and to reattach (to view progress, for instance) one can use:

```
screen -r pca_generation
```

A word of warning - this script does not currently actively save data - it only saves at the end. So, try not to kill it before it's done!


EDIT: This code uses MPI rather than multiprocessing, so to utilize all the cores on one's machine, they should do the following:
```
mpiexec --map-by ppr:n:socket python pauli_channel_approximation.py
```

Where the n in ppr:socket is the number of cores to parallelize over.
