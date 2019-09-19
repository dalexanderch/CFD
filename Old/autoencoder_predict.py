from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

# arguments
imgPath = sys.argv[1]
print('Image path : {}'.format(imgPath))

# load image
image = Image.open(imgPath)
image = np.asarray(image)
image = image.flatten()
image = image.astype('float32') / 255

# load model
autoencoder = load_model('autoencoder.h5')
autoencoder.summary()

image = np.array([image])
# make prediction
predicted_img = autoencoder.predict(image, verbose=0)

# save prediction
predicted_img = predicted_img.reshape(56,56)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predicted_img.gif', format = 'GIF')


