from sklearn.metrics import mean_squared_error
from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, UpSampling2D
from keras.models import Model
import sys

# parameters 
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255


# Resize the data
x_train_small = []
x_test_small = []
for image in x_train:
	image = Image.fromarray(image, mode='L')
	image = image.resize((14,14), resample=Image.BILINEAR)
	x_train_small.append(np.asarray(image))

x_train_small = np.array(x_train_small)

for image in x_test:
	image = Image.fromarray(image, mode='L')
	image = image.resize((14,14), resample=Image.BILINEAR)
	x_test_small.append(np.asarray(image))

x_test_small = np.array(x_test_small)


# save arrays for image in x_train: 
# np.save('x_train', x_train)
# np.save('x_test', x_test)
# np.save('x_train_big', x_train_big)
# np.save('x_test_big', x_test_big)

print("Successfuly generated appropriate data")

# prepare array for tensorflow
x_train = np.expand_dims(x_train, x_train.shape[-1])
x_train_small = np.expand_dims(x_train_small, x_train_small.shape[-1])
x_test = np.expand_dims(x_test, x_test.shape[-1])
x_test_small = np.expand_dims(x_test_small, x_test_small.shape[-1])

# Build model
input_img = Input(shape=(x_train_small.shape[1], x_train_small.shape[2], 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)
x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)


upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

#Train the model
upsample.fit(x_train_small, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test_small, x_test))



#################### Save model
upsample.save("upsample.h5")