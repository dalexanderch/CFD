from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

#input arguments 
latent_dim = int(sys.argv[1])
epochs = int(sys.argv[2])

# Prepare model 

# size of latent vector.
encoding_dim = latent_dim

# size of flattened input vector (eg: 32*32 image gives dim 2500)
input_dim = 784

# size of flattened ouput vector
output_dim = 3136

# input placeholder
input_img = Input(shape=(input_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)


# "decoded" is the lossy reconstruction of the input
decoded = Dense(output_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded  input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# compile the model, optimizer is adam, loss is MSE
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



# load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()


# resize the images
x_train = tf.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
y_train = tf.image.resize_images(x_train, (56,56))
x_test = tf.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
y_test = tf.image.resize_images(x_test, (56,56))


#flatten the images
x_train = tf.reshape(x_train,(x_train.shape[0],x_train.shape[1] * x_train.shape[2]))
x_test = tf.reshape(x_test,(x_test.shape[0],x_test.shape[1] * x_test.shape[2]))
y_train = tf.reshape(y_train,(y_train.shape[0],y_train.shape[1] * y_train.shape[2]))
y_test = tf.reshape(y_test,(y_test.shape[0],y_test.shape[1] * y_test.shape[2]))
sess = tf.Session()
x_train = x_train.eval(session=sess)
y_train = y_train.eval(session=sess)
y_test = y_test.eval(session=sess)
x_test = x_test.eval(session=sess)
sess.close()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)
print ('y_train shape:', y_train.shape)
print ('y_test shape:', y_test.shape)

# train the model
autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, y_test))


# evaluate the model
scores = autoencoder.evaluate(x_train, y_train, verbose=0)
print('{} : {}'.format(autoencoder.metrics_names[0], scores))

# save model 
autoencoder.save("autoencoder.h5")

img_large = x_test[1]
img_large = img_large.reshape(28,28)
img_large = img_large.astype('uint8')
img_large = Image.fromarray(img_large * 255, mode='L')
img_large.save('img1.gif', format = 'GIF')




