from PIL import Image
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import os

# Save original
curr = os.getcwd()
image1 = Image.open(curr + "/data/img/000001.jpg")
image1 = image.convert('L')
image1.save('img1.jpg', 'JPEG')
image2 = Image.open(curr + "/data/img/000002.jpg")
image2 = image.convert('L')
image2.save('img2.jpg', 'JPEG')
image3 = Image.open(curr + "/data/img/000003.jpg")
image3 = image.convert('L')
image3.save('img3.jpg', 'JPEG')
image4 = Image.open(curr + "/data/img/000004.jpg")
image4 = image.convert('L')
image4.save('img4.jpg', 'JPEG')

# Resize
image1 = image1.resize((109,89), resample=Image.BILINEAR)
image2 = image2.resize((109,89), resample=Image.BILINEAR)
image3 = image3.resize((109,89), resample=Image.BILINEAR)
image4 = image4.resize((109,89), resample=Image.BILINEAR)

# Predict
upsample = load_model('upclassic.h5')
predicted_img = np.asarray(image1)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img1classic.jpg', 'JPEG')

predicted_img = np.asarray(image2)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img2classic.jpg', 'JPEG')

predicted_img = np.asarray(image3)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img3classic.jpg', 'JPEG')

predicted_img = np.asarray(image4s)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img4classic.jpg', 'JPEG')

# Predict
upsample = load_model('upsamples.h5')
predicted_img = np.asarray(image1)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img1model.jpg', 'JPEG')

predicted_img = np.asarray(image2)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img2model.jpg', 'JPEG')

predicted_img = np.asarray(image3)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img3model.jpg', 'JPEG')

predicted_img = np.asarray(image4)
predicted_img = predicted_img/255
predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
predicted_img = upsample.predict(predicted_img)
predicted_img = 255 * predicted_img
predicted_img = predicted_img.astype('int8')
predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
predicted_img = Image.fromarray(predicted_img, mode='L')
predicted_img = predicted_img.resize((178,218), resample=Image.BILINEAR)
predicted_img.save('img4model.jpg', 'JPEG')
