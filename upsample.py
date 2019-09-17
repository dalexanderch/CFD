from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import sys


#################### Parameters
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

#################### Load data
def load_data( path ): # path with respect to current directy
	cur_dir = os.getcwd()
	path = cur_dir + path
	print(path)
	images = sorted(os.listdir(path)) 

	data = []
	for filename in images:
		image = Image.open(path + filename)
		image = np.asarray(image)
		image = image.astype('float32') / 255
		data.append(np.asarray(image)) 

	data = np.array(data)
	return data

# load small images
small = load_data("/data/small/")

# load big images 
big = load_data("/data/big/")

# split into training set and validation set 
x_train, x_test, y_train, y_test = train_test_split(small, big, test_size=0.2) # We take 20% of samples for validation

# adapt for use with tf
x_train = np.expand_dims(x_train[:, 0:x_train.shape[1] - 1, 0:x_train.shape[2] - 1], axis=3)
x_test = np.expand_dims(x_test[:, 0:x_test.shape[1] - 1, 0:x_test.shape[2] - 1], axis=3)
y_train = np.expand_dims(y_train[:, 0:y_train.shape[1] - 1, 0:y_train.shape[2] - 1], axis=3)
y_test = np.expand_dims(y_test[:, 0:y_test.shape[1] - 1, 0:y_test.shape[2] - 1], axis=3)

#################### Build model
input_img = Input(shape=(x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2))(input_img)
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)


upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='binary_crossentropy')
upsample.summary()

#################### Train the model
upsample.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, y_test))



#################### Save model
upsample.save("upsample.h5")





