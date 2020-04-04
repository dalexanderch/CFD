 #!/usr/bin/env python -W ignore::FutureWarning

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import glob
import os
import re
import numpy as np
import math
from keras.layers import Input, UpSampling2D
from keras.models import Model

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Load small 
print("Loading small data")
path = os.getcwd() + "/small"
files = [f for f in glob.glob(path + "**/*.dat")]
files = sorted_nicely(files)
x = []
for file in files:
    data = np.loadtxt(file, dtype = np.float32)
    data = np.resize(data, (101, 41))
    data = data.transpose()
    x.append(data)
    
x = np.array(x)

# Load big
print("Loading big data")
path = os.getcwd() + "/big"
files = [f for f in glob.glob(path + "**/*.dat")]
files = sorted_nicely(files)
y = []
for file in files:
    data = np.loadtxt(file, dtype=np.float32)
    data = np.resize(data, (202, 82))
    data = data.transpose()
    y.append(data)
    
y = np.array(y)

print("Loaded big data")
# Split into training and test set
n = math.floor(x.shape[0]/10)
x_test = x[0:n, :, :]
y_test = y[0:n, :, :]
x_train= x[n+1:-1, :, :]
y_train= y[n+1:-1, :, :]

# Prepare data
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
y_train = np.expand_dims(y_train, axis = 3)
y_test = np.expand_dims(y_test, axis = 3)

# Build model
input_img = Input(shape=(x_train.shape[1], x_train.shape[2], 1)) 
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

#Train the model
upsample.fit(x_train, y_train,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, y_test))

# Evaluate
print(upsample.evaluate(x_test, y_test))
