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
            first = False
        elif (update_x == True and update_y == False):
            X = it1.next()
            yield X,Y
            update_x = False
            update_y = True
        elif (update_x == False and update_y == True):
            Y = it2.next()
            yield X,Y
            update_x = True
            update_y = False



# Define our custom metric
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
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
upsample = Model(input_img, x)

upsample.compile(optimizer='adadelta', loss='mean_squared_error', metrics=[PSNR])

# Train
g_train = gen(train_small_it, train_it)
g_val = gen(val_small_it, val_it)

upsample.fit_generator(
	generator = g_train,
	steps_per_epoch = math.ceil(180000/batch_size),
	epochs = epochs,
	validation_data = g_val,
	validation_steps = math.ceil(18000/batch_size),
	use_multiprocessing=True
	)

# Save weights
upsample.save("dual_model.h5")

# Evaluate
print(upsample.evaluate_generator(generator = g_val, steps=18000, use_multiprocessing=True))
