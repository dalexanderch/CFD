from sklearn.metrics import mean_squared_error
from PIL import Image
import numpy as np
from keras.datasets import cifar100
from keras.layers import Input, Dense, Conv2D, UpSampling2D
from keras.models import Model
import sys
import math
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
# x1 y1 x2 y1 x2 y2 x3 y2 x3 y3
def gen(it1, it2):
    update_x = True
    update_y = True
    first = True
    while True:
        if first:
            X = it1.next()
            Y = it2.next()
            yield X,Y
            update_x = True
            update_y = False
        elif update_x == True && update_y == False:
            X = it1.next()
            yield X,Y
            update_x = False
            update_y = True
        elif update_x == False && update_y = True:
            Y = it2.next()
            yield X,Y
            update_x = True
            update_y = False

def dual(y_true,y_pred):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return mse


# class DualLoss:
#   def __init__(self):
#     self.var = None
#
#   def __call__(self, y_true, y_pred, sample_weight=None):
#     mse = K.mean(K.square(y_true - y_pred), axis=-1)
#     if self.var is None:
#         z = np.zeros((200,80))
#         self.var = K.variable(z)
#         return mse
#     mseprev = K.mean(K.square(self.var - y_pred), axis=-1)
#     self.var = K.update(self.var, y_true)
#     return (mse + mseprev)/2

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))

# parameters
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# Create generator
datagen = ImageDataGenerator(validation_split=0.1, rescale=1./255)

# Prepare training and validation  datasets
train_it = datagen.flow_from_directory(directory='data/big/', target_size=(200,80), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training')
val_it = datagen.flow_from_directory(directory='data/big/', target_size=(200,80), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation')
train_small_it  = datagen.flow_from_directory(directory='data/small/', target_size=(100,40), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training', interpolation = 'bilinear')
val_small_it =  datagen.flow_from_directory(directory='data/small/', target_size=(100,40), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation', interpolation = 'bilinear')

# Build model
input_img = Input(shape=(100, 40, 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
upsample = Model(input_img, x)

upsample.compile(optimizer='adadelta', loss=dual, metrics=[PSNR])

# Train
g_train = gen(train_small_it, train_it)
g_val = gen(val_small_it, val_it)

upsample.fit_generator(
	generator = g_train,
	steps_per_epoch = math.ceil(45000/batch_size),
	epochs = epochs,
	validation_data = g_val,
	validation_steps = math.ceil(4500/batch_size),
	use_multiprocessing=True
	)

# Save weights
# upsample.save("upsample.h5")

# Evaluate
print(upsample.evaluate_generator(generator = g_val, steps=634, use_multiprocessing=True))
