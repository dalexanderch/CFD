#!/usr/bin/env python -W ignore::FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import glob
import os
import math
from keras.layers import Input, UpSampling2D, Conv2D, BatchNormalization
from keras.models import Model
from sequence import data
import sys
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf



# Define our custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))

def ssim(y_true, y_pred):
  return - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))  
  
# Constants
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# Compute steps per epochs
path = os.getcwd() + "/small"
files_x = [f for f in glob.glob(path + "**/*.npy")]
path = os.getcwd() + "/big"
files_y = [f for f in glob.glob(path + "**/*.npy")]
x_train, x_val, y_train, y_val = train_test_split(files_x, files_y, test_size=0.1)

# Build Sequence
seq_train = data(x_train, y_train, batch_size)
seq_val =  data(x_val, y_val, batch_size)

# Parameters
steps_per_epoch = math.floor(len(x_train)/batch_size)
validation_steps = math.floor(len(x_val)/batch_size) 

# Build model
input_img = Input(shape=(41, 101, 1)) 
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error', metrics=[PSNR, ssim])

# Save the model
plot_model(upsample,show_shapes=True, to_file='model_large.png')

#Train the model
upsample.fit_generator(generator = seq_train,
                steps_per_epoch=steps_per_epoch,
                validation_data = seq_val,
                validation_steps = validation_steps,
                epochs = epochs,
                shuffle=True,
                workers=8,
                max_queue_size=10,
                use_multiprocessing = True
                )

# Save weights
upsample.save("model.h5")

# Evaluate
print(upsample.evaluate_generator(generator = seq_val, steps=validation_steps, use_multiprocessing=True))