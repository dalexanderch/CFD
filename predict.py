from keras.datasets import mnist
from keras.models import load_model
from PIL import Image
import os
import numpy as np

# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)

# Build path
cur_dir = os.getcwd()
pathoriginal = "/data/mnist/"
pathoriginal = cur_dir + pathoriginal
print(pathoriginal)
# Save first thousand
for id, image in enumerate(x_train[0:1000,:,:]):
	image = image.astype("uint8")
	image = Image.fromarray(image, mode='L')
	image.save(pathoriginal + "{}.gif".format(id), format = 'GIF')

# Load model 
autoencoder = load_model('autoencoder.h5')
autoencoder.summary()
	
# Predict and save
images = sorted(os.listdir(pathoriginal)) 
pathsave = "/data/pred/"
pathsave = cur_dir + pathsave
for id, filename in enumerate(images):
	image = Image.open(pathoriginal + filename)
	image = np.asarray(image)
	image = image.flatten()
	image = image.astype('float32') / 255
	image = np.array([image])
	predicted_img = autoencoder.predict(image, verbose=0)
	# save predicted_image
	predicted_img = predicted_img.reshape(56,56)
	predicted_img = 255 * predicted_img
	predicted_img = predicted_img.astype('int8')
	predicted_img = Image.fromarray(predicted_img, mode='L')
	predicted_img.save(pathsave + "{}.gif".format(id), format = 'GIF')