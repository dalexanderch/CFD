import numpy as np
import subprocess
import os

# Constants
N = 100
lx = 2
ly = 1
Re = 600
numSamples = 1
folder = "./data/tmp"

# Generate random Reynolds number (sample from normal distribution with mean Re)
mu, sigma = Re, 20       # mean and standard deviation
samples = np.random.normal(mu, sigma, numSamples)
print(samples)

# Generate the data
args = ("./cylinder2d", N, Re, lx, ly, numSamples)
# build the folder if doesn't exist
cmd = "mkdir" + folder
print(cmd)
# popen = subprocess.Popen(args, stdout=subprocess.PIPE)
# popen.wait()
# output = popen.stdout.read()
# print (output)
