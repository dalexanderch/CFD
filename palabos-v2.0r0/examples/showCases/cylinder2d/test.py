#!/usr/bin/env python -W ignore::FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import glob
import os
import math
from keras.layers import Input, UpSampling2D, Conv2D
from keras.models import Model
from generator import image_generator
import sys

# Constants
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# Compute steps per epochs
path = os.getcwd() + "/small"
files = [f for f in glob.glob(path + "**/*.dat")]
steps_per_epoch = math.floor(len(files)/batch_size)

# Build generator
g = image_generator(len(files), 32)

# Build model
input_img = Input(shape=(41, 101, 1)) 
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

#Train the model
upsample.fit_generator(g,
                steps_per_epoch=steps_per_epoch,
                epochs = epochs,
                shuffle=True,
                workers=8,
                max_queue_size=10,
                )