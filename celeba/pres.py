from PIL import Image
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import os

# Save original
curr = os.getcwd()
image = Image.open(curr + "/data/img/000001.jpg")
image = image.convert('L')
image.save('original.jpg', 'JPEG')

# Resize
image = image.resize((109,89), resample=Image.BILINEAR)

# Predict
upsample = load_model('upsample.h5')
predicted_img = np.asarray(image)
print(predicted_img.shape)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
print(predicted_img.shape)
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('predicted.jpg', 'JPEG')


