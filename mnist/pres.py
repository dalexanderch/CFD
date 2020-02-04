from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras.models import load_model


# load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()

# Save original
img1 = Image.fromarray(x_test[0], 'L')
img1.save('img1.png', 'PNG')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255

x_test_small = []

for image in x_test:
	image = Image.fromarray(image, mode='F')
	image = image.resize((14,14), resample=Image.BILINEAR)
	x_test_small.append(np.asarray(image))

x_test_small = np.array(x_test_small)

# Load model and predict
upsample = load_model('upsample.h5')

image = x_test_small[0]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predicted.png', 'PNG')