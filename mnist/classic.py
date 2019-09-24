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
x_train_big = []
x_test_big = []
for image in x_train:
	image = Image.fromarray(image, mode='L')
	image = image.resize((56,56), resample=Image.BILINEAR)
	x_train_big.append(np.asarray(image))

x_train_big = np.array(x_train_big)

for image in x_test:
	image = Image.fromarray(image, mode='L')
	image = image.resize((56,56), resample=Image.BILINEAR)
	x_test_big.append(np.asarray(image))

x_test_big = np.array(x_test_big)


# save arrays for image in x_train: 
# np.save('x_train', x_train)
# np.save('x_test', x_test)
# np.save('x_train_big', x_train_big)
# np.save('x_test_big', x_test_big)

print("Successfuly generated appropriate data")

# prepare array for tensorflow
x_train = np.expand_dims(x_train, x_train.shape[-1])
x_train_big = np.expand_dims(x_train_big, x_train_big.shape[-1])
x_test = np.expand_dims(x_test, x_test.shape[-1])
x_test_big = np.expand_dims(x_test_big, x_test_big.shape[-1])

# Build model
input_img = Input(shape=(x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format
x = UpSampling2D((2, 2), interpolation='bilinear')(input_img)



upsample = Model(input_img, x)
upsample.compile(optimizer='adadelta', loss='mean_squared_error')

#Train the model
upsample.fit(x_train, x_train_big,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test_big))



#################### Save model
upsample.save("upsample.h5")