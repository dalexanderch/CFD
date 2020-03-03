from PIL import Image
import numpy as np
from keras.models import load_model
import math
from keras import backend as K
import sys
import os


# Define our custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))

# Save original
curr = os.getcwd()
imgs = []
for i in range(1, 500):
    img = Image.open(curr + "/data/small/small/{}.jpg".format(i))
    img = img.convert('L')
    img.save(curr + '/original/img{}.gif'.format(i), 'GIF')
    imgs.append(img)


# Resize
for i,img in enumerate(imgs):
    imgs[i] = img.resize((40,100), resample=Image.BILINEAR)



# Predict
dependencies = {
     'PSNR': PSNR
}

# Load model and predict
upsample = load_model('upclassic.h5', custom_objects=dependencies)
for i,img in enumerate(imgs):
    predicted_img = np.asarray(img)
    predicted_img = predicted_img/255.00
    predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
    predicted_img = upsample.predict(predicted_img)
    predicted_img = 255 * predicted_img
    predicted_img = predicted_img.astype('int8')
    predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
    predicted_img = Image.fromarray(predicted_img, mode='L')
    predicted_img = predicted_img.resize((200,80), resample=Image.BILINEAR)
    predicted_img.save(curr + '/classic/{}.gif'.format(i), 'GIF')

# Load model and predict
upsample = load_model('upsample.h5', custom_objects=dependencies)
for i,img in enumerate(imgs):
    predicted_img = np.asarray(img)
    predicted_img = predicted_img/255.00
    predicted_img = predicted_img.reshape(1, predicted_img.shape[0], predicted_img.shape[1], 1 )
    predicted_img = upsample.predict(predicted_img)
    predicted_img = 255 * predicted_img
    predicted_img = predicted_img.astype('int8')
    predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])
    predicted_img = Image.fromarray(predicted_img, mode='L')
    predicted_img = predicted_img.resize((200,80), resample=Image.BILINEAR)
    predicted_img.save(curr + '/model/{}.gif'.format(i), 'GIF')
