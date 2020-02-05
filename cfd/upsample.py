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
train_it = datagen.flow_from_directory(directory='datacfd/big/', target_size=(200,80), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training')
val_it = datagen.flow_from_directory(directory='datacfd/big/', target_size=(200,80), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation')
train_small_it  = datagen.flow_from_directory(directory='datacfd/small/', target_size=(100,40), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='training', interpolation = 'bilinear')
val_small_it =  datagen.flow_from_directory(directory='datacfd/small/', target_size=(100,40), shuffle=False, color_mode='grayscale', class_mode=None, batch_size=batch_size, subset='validation', interpolation = 'bilinear')

# Build model
input_img = Input(shape=(100, 40, 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

# Train
g_train = gen(train_small_it, train_it)
g_val = gen(val_small_it, val_it)

upsample.fit_generator(
	generator = g_train,
	steps_per_epoch = 5727, # 183240/32 rounded upward
	epochs = 10,
	validation_data = g_val,
	validation_steps = 634, # 20259/256 rounded upward
	use_multiprocessing=True
	)

# Save weights
upsample.save("upsample.h5")

# Evaluate
print(upsample.evaluate_generator(generator = g_val, steps=634, use_multiprocessing=True))