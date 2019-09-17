from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#################### Load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[0:2000]
x_test = x_test[0:200]

# resize the images
x_train = tf.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
y_train = tf.image.resize_images(x_train, (56,56))
x_test = tf.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
y_test = tf.image.resize_images(x_test, (56,56))

# turn to numpy array
sess = tf.Session()
x_train = x_train.eval(session=sess)
y_train = y_train.eval(session=sess)
y_test = y_test.eval(session=sess)
x_test = x_test.eval(session=sess)
sess.close()

# normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)
print ('y_train shape:', y_train.shape)
print ('y_test shape:', y_test.shape)

#################### Build model
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)


autoencoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

#################### Train
autoencoder.fit(x_train, y_train,
                epochs=5,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, y_test))


#################### Save model
autoencoder.save("upsample.h5")




decoded_imgs = autoencoder.predict(x_test)

fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(decoded_imgs[0].reshape(56,56))
fig.add_subplot(2, 2, 2)
plt.imshow(decoded_imgs[1].reshape(56,56))
fig.add_subplot(2, 2, 3)
plt.imshow(x_test[0].reshape(28,28))
fig.add_subplot(2, 2, 4)
plt.imshow(x_test[1].reshape(28,28))
plt.show()
