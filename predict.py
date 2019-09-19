from keras.datasets import mnist
from keras.models import load_model
from PIL import Image
import os
import numpy as np
import sys

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


# Parameters
pathSmall = sys.argv[1]
pathBig = sys.argv[2]

# Build path
cur = os.getcwd()
print("Path of original images (with respect to current path) : {}".format(pathSmall))
print("Path to save images : (with respected to current path) {}".format(pathBig))

# Load small images 
small = load_data(pathSmall)
print(small.shape)

# Load model 
upsample = load_model('upsample.h5')
upsample.summary()

# Predict and save
for i, image in enumerate(small):
	image = image[0:image.shape[0]-1, 0:image.shape[1] - 1]
	image = image.reshape(1, image.shape[0], image.shape[1], 1 )
	predicted_img = upsample.predict(image, verbose=0)
	predicted_img = 255 * predicted_img
	predicted_img = predicted_img.astype('int8')
	predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
	predicted_img = Image.fromarray(predicted_img, mode='L')
	predicted_img.save(cur + pathBig + "{}.gif".format(i), format = 'GIF')
	print("Saved image {}".format(i))
	# save 
# # Build path
# cur_dir = os.getcwd()
# pathoriginal = "/data/mnist/"
# pathoriginal = cur_dir + pathoriginal
# print(pathoriginal)
# # Save first thousand
# for id, image in enumerate(x_train[0:1000,:,:]):
# 	image = image.astype("uint8")
# 	image = Image.fromarray(image, mode='L')
# 	image.save(pathoriginal + "{}.gif".format(id), format = 'GIF')

# # Load model 
# autoencoder = load_model('autoencoder.h5')
# autoencoder.summary()
	
# # Predict and save
# images = sorted(os.listdir(pathoriginal)) 
# pathsave = "/data/pred/"
# pathsave = cur_dir + pathsave
# for id, filename in enumerate(images):
# 	image = Image.open(pathoriginal + filename)
# 	image = np.asarray(image)
# 	image = image.flatten()
# 	image = image.astype('float32') / 255
# 	image = np.array([image])
# 	predicted_img = autoencoder.predict(image, verbose=0)
# 	# save predicted_image
# 	predicted_img = predicted_img.reshape(56,56)
# 	predicted_img = 255 * predicted_img
# 	predicted_img = predicted_img.astype('int8')
# 	predicted_img = Image.fromarray(predicted_img, mode='L')
# 	predicted_img.save(pathsave + "{}.gif".format(id), format = 'GIF')