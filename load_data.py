import numpy as np
from PIL import Image
import cv2
import os
import os.path
from sklearn.model_selection import train_test_split


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



small = load_data("/data/small/")
small.reshape(small.shape[0], small.shape[1], small.shape[2], 1)
sm = small[:, 0:100, 0:100]
print(sm.shape)
big = load_data("/data/big/")
x_train, x_test, y_train, y_test = train_test_split(small, big, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
