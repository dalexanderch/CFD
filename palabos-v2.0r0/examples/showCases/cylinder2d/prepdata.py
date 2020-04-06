import os
import numpy as np
import re
import glob
from tqdm import tqdm

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


# Load small 
path = os.getcwd() + "/small"
files = [f for f in glob.glob(path + "/*.dat")]
files = sorted_nicely(files)
print("Prepping {} small data".format(len(files)))
for file in tqdm(files):
    data = np.loadtxt(file)
    data = np.resize(data, (101, 41))
    data = data.transpose()
    np.save(os.path.splitext(file)[0], data)
    os.remove(file)
    

# Load big
path = os.getcwd() + "/big"
files = [f for f in glob.glob(path + "/*.dat")]
files = sorted_nicely(files)
print("Prepping {} big data".format(len(files)))
for file in tqdm(files):
    data = np.loadtxt(file)
    data = np.resize(data, (201, 81))
    data = data.transpose()
    data = np.append(data, np.zeros((1, 201)), axis = 0)
    data = np.append(data, np.zeros((81, 1)), axis = 1)
    np.save(os.path.splitext(file)[0], data)
    os.remove(file)