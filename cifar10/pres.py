from PIL import Image
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model


# load MNIST data
(x_train, _), (x_test, _) = cifar10.load_data()

x_test_tmp = []
for image in x_test:
	image = image * 255
	image = image.astype('uint8')
	image = Image.fromarray(image, mode = 'RGB')
	image = image.convert('L')
	image = np.asarray(image)
	x_test_tmp.append(image)

x_test = np.array(x_test_tmp)

x_test = x_test.astype('float32') / 255


# Save original
img1 = x_test[0]
img1 = img1 * 255
img1 = img1.astype('uint8')
img1 = Image.fromarray(img1, mode = 'L')
img1.save('original.png', 'PNG')


x_test_small = []

for image in x_test:
	image = Image.fromarray(image, mode='F')
	image = image.resize((16,16), resample=Image.BILINEAR)
	x_test_small.append(np.asarray(image))

x_test_small = np.array(x_test_small)

# Load model and predict
upsample = load_model('upsample.h5')

image = x_test_small[0]
print(image)
image = image[0:image.shape[0], 0:image.shape[1]]
image = image.reshape(1, image.shape[0], image.shape[1], 1 )
predicted_img = upsample.predict(image)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
print(predicted_img)
predicted_img = Image.fromarray(predicted_img, mode='L')

predicted_img.save('predicted.png', 'PNG')