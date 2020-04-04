import os
import numpy as np
import glob
import time

path = os.getcwd() + "/big/1.dat"
start = time.time()
data = np.loadtxt(path, dtype=np.float32)
print(data.dtype)
np.savetxt(path, data)
data = np.loadtxt(path)
print(data.dtype)