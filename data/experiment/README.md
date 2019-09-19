The rb_data.npy file contains a 6,6,10 numpy array. The first index indexes over the pulse that was used - the first pulse is the calibrated one, 2-5 are over and under calibrated, and the last one is the one from our routine - sampling from the over and under calibrated pulses. 

The second index indexes over the sequence lengths - 2, 4, 8, 16, 32, and 64.

The last index indexes over different Clifford sequences of the given length. I drew 10 at random (and I did not save those sequences here.). However, it is the same random sequences for each pulse definition.
