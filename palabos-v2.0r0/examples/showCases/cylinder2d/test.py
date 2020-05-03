import os
import numpy as np

path = os.getcwd() + "/small/1.npy"

x = np.load(path)
m = np.amax(x)