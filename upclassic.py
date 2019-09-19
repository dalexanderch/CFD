import numpy as np
from PIL import Image
import os
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

def mse(img1,img2):
	if img1.shape != img2.shape:
		return -1
	else:
		mse = 0;
		for i in range(0, img1.shape[0]):
			for j in range(0, img2.shape[]):
				mse = mse + (img1[i,j] - img2[i,j])^2
		mse = mse / (img1.shape[0] * img2.shape[1])
		return mse




# load small images
small = load_data("/data/small/")

# load big images 
big = load_data("/data/big/")

# Resize small using pillow
resized = []
for i, image in enumerate(small):
	N = 2 * image.shape[0]
	M = 2 * image.shape[1]
	image = Image.fromarray(image[0:N/2 -1, 0:N/2 - 1], mode='F')
	image = image.resize((N, M), Image.BILINEAR)
	resized.append(image)

# Compute mse
mse = 0
for img1 in resized:
	for image2 in big:
		mse = mse + mse(img1,img2)
resized = np.array(resized)
