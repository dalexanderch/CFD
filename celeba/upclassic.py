from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from PIL import Image
import numpy as np
from keras.datasets import cifar100
from keras.layers import Input, Dense, Conv2D, UpSampling2D
from keras.models import Model
import sys
import math
from keras import backend as K


def gen(it1, it2):
    while True:
        X = it1.next()
        Y = it2.next()
        yield X,Y

# Define our custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))

# Parameters
batch_size = int(sys.argv[1])

# Create generator
datagen = ImageDataGenerator(validation_split=0.1, rescale=1./255)

# Prepare training and validation  datasets
train_it = datagen.flow_from_directory(directory='data/databig', target_size=(178,218), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training')
val_it = datagen.flow_from_directory(directory='data/databig/', target_size=(178,218), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation')
train_small_it  = datagen.flow_from_directory(directory='data/datasmall', target_size=(89,109), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training', interpolation = 'bilinear')
val_small_it =  datagen.flow_from_directory(directory='data/datasmall', target_size=(89,109), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation', interpolation = 'bilinear')

# Build model
input_img = Input(shape=(89, 109, 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)

upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error', metrics=[PSNR])

# Train
g_train = gen(train_small_it, train_it)
g_val = gen(val_small_it, val_it)

upsample.fit_generator(
	generator = g_train,
	steps_per_epoch = 715, # 183240/32 rounded upward
	epochs = 1,
	validation_data = g_val,
	validation_steps = 79, # 20259/256 rounded upward
	use_multiprocessing=True
	)

# Save weights
upsample.save("upclassic.h5")

# Evaluate
print(upsample.evaluate_generator(generator = g_val, steps=79, use_multiprocessing=True))
