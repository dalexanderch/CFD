from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import math
from keras import backend as K
import sys

# Define our custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))

# load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()

# Save original
img1 = Image.fromarray(x_test[0], 'L')
img1.save('img1.png', 'PNG')
img2 = Image.fromarray(x_test[1], 'L')
img2.save('img2.png', 'PNG')
img3 = Image.fromarray(x_test[2], 'L')
img3.save('img3.png', 'PNG')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255

x_test_small = []

for image in x_test:
	image = Image.fromarray(image, mode='F')
	image = image.resize((14,14), resample=Image.BILINEAR)
	x_test_small.append(np.asarray(image))

x_test_small = np.array(x_test_small)

dependencies = {
     'PSNR': PSNR
}

# Load model and predict
upsample = load_model('upclassic.h5', custom_objects=dependencies)


image = x_test_small[0]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedclassic1.png', 'PNG')

image = x_test_small[1]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedclassic2.png', 'PNG')

image = x_test_small[2]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedclassic3.png', 'PNG')

dependencies = {
     'PSNR': PSNR
}

# Load model and predict
upsample = load_model('upsample.h5', custom_objects=dependencies)

image = x_test_small[0]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedmodel1.png', 'PNG')

image = x_test_small[1]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedmodel2.png', 'PNG')

image = x_test_small[2]
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img.save('predictedmodel3.png', 'PNG')
