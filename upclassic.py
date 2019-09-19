# import numpy as np
# from PIL import Image
# import os
# import sys

# def load_data( path ): # path with respect to current directy
# 	cur_dir = os.getcwd()
# 	path = cur_dir + path
# 	images = sorted(os.listdir(path)) 

# 	data = []
# 	for filename in images:
# 		image = Image.open(path + filename)
# 		image = np.asarray(image)
# 		image = image.astype('float32') / 255
# 		data.append(np.asarray(image)) 

# 	data = np.array(data)
# 	return data

# def mse(img1,img2):
# 	if img1.shape != img2.shape:
# 		return -1
# 	else:
# 		mse = 0;
# 		for i in range(0, img1.shape[0]):
# 			for j in range(0, img2.shape[1]):
# 				mse = mse + (img1[i,j] - img2[i,j])^2
# 		mse = mse / (img1.shape[0] * img2.shape[1])
# 		print(mse)
# 		return mse




# # load small images
# small = load_data("/data/small/")

# # load big images 
# big = load_data("/data/big/")
# tmp = big
# big = []
# # Resize big
# for i, image in enumerate(tmp):
# 	image = image[0:200,0:200]
# 	big.append(image)

# big = np.array(big)
# print(big.shape)

# # Resize small using pillow
# resized = []
# for i, image in enumerate(small):
# 	image = Image.fromarray(image[0:99, 0:99], mode='F')
# 	image = image.resize((200, 200), Image.BILINEAR)
# 	image = np.asarray(image)
# 	resized.append(image)

# resized = np.array(resized)


# # Compute mse
# err = 0
# for i in range(0, resized.shape[0]):
# 	err = err + mse(resized[i], big[i])
# 	print(err)
# err = err / (resized.shape[0])
# print(err)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import sys
from keras import backend as K


#################### Parameters
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF")


#################### Load data
def load_data( path ): # path with respect to current directy
	cur_dir = os.getcwd()
	path = cur_dir + path
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
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)


upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')
upsample.summary()

#################### Train the model
upsample.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, y_test))



#################### Save model
upsample.save("upclassic.h5")





