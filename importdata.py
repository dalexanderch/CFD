from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
import glob

######### load images
# create generator
datagen = ImageDataGenerator(rescale=1./255)
# prepare an iterators for each dataset
x_train_it1 = datagen.flow_from_directory('data/train/',
    target_size=(100, 100),
    color_mode="grayscale",
    batch_size=5,
    shuffle=False,
    class_mode=None)

x_train_it2 = datagen.flow_from_directory('data/train/',
    target_size=(100, 100),
    color_mode="grayscale",
    batch_size=5,
    shuffle=False,
    class_mode=None)

def gen(it1,it2):
    while True:
        X = it1.next()
        Y = it2.next()
        yield X,Y

# # confirm the iterator works
# batchX = x_train_it1.next()
# plt.imshow(batchX[0].reshape(100,100))
# plt.show()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



######### build model
input_img = Input(shape=(100,100,1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
g = gen(x_train_it1,x_train_it2)
autoencoder.fit_generator(
        g,
        steps_per_epoch=100,
        epochs=5)

predict = autoencoder.predict_generator(g,1)
fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(predict[0].reshape(100,100))
fig.add_subplot(1, 2, 2)
plt.imshow(predict[1].reshape(100,100))
plt.show()
