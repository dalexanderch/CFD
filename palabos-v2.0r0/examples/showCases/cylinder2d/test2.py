import os
import numpy as np
import glob
import time

path = os.getcwd() + "/big/1.dat"
start = time.time()
data = np.loadtxt(path)
elapsed = time.time() - start
print(elapsed)
data = np.resize(data, (101, 41))
data = data.transpose()
elapsed = time.time() - start
print(elapsed)