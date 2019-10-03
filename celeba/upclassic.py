from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.layers import Input, Dense, Conv2D, UpSampling2D
from keras.models import Model
import numpy as np

def gen(it1, it2):
    while True:
        X = it1.next()
        Y = it2.next()
        yield X,Y

# Parameters 
batch_size = 32
epochs = 20

# Create generator
datagen = ImageDataGenerator(validation_split=0.1, rescale=1./255)

# Prepare training and validation  datasets
train_it = datagen.flow_from_directory(directory='data/big/', target_size=(178,218), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training')
val_it = datagen.flow_from_directory(directory='data/big/', target_size=(178,218), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation')
train_small_it  = datagen.flow_from_directory(directory='data/small/', target_size=(89,109), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training')
val_small_it =  datagen.flow_from_directory(directory='data/small/', target_size=(89,109), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation')

# Build model
input_img = Input(shape=(89, 109, 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)

upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

# Train
g_train = gen(train_small_it, train_it)
g_val = gen(val_small_it, val_it)

upsample.fit_generator(
	generator = g_train,
	steps_per_epoch = 5727, # 183240/32 rounded upward
	epochs = 1,
	validation_data = g_val,
	validation_steps = 634 # 20259/256 rounded upward
	)

# Save weights
upsample.save("upsample.h5")

# Evaluate
print(upsample.evaluate_generator(generator = g_train))