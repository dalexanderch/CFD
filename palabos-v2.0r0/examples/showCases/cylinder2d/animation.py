import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import glob
import re
import matplotlib.animation as animation
from keras import backend as K
import math

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Define our custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))
    
dependencies = {
     'PSNR': PSNR
}

# Prepare data
path = os.getcwd() + "/big"
files = [f for f in glob.glob(path + "**/*.npy")]
files = sorted_nicely(files)

data = []
for file in files:
    tmp = np.load(file)
    data.append(tmp)

fig = plt.figure()
ims = []
for d in data:
    im = plt.imshow(d, animated=True)
    ims.append([im])
#Show
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('original.mp4')

# Load model
upsample = load_model("classic.h5", custom_objects=dependencies)
# Prepare data
path = os.getcwd() + "/small"
files = [f for f in glob.glob(path + "**/*.npy")]
files = sorted_nicely(files)

data = []
for file in files:
    tmp = np.load(file)
    tmp = np.reshape(tmp, (1,41,101,1))
    tmp = upsample.predict(tmp)
    tmp = np.reshape(tmp, (82,202))
    data.append(tmp)

fig = plt.figure()
ims = []
for d in data:
    im = plt.imshow(d, animated=True)
    ims.append([im])
#Show
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('classic.mp4')

# Load model
upsample = load_model("model.h5", )
# Prepare data
path = os.getcwd() + "/small"
files = [f for f in glob.glob(path + "**/*.npy")]
files = sorted_nicely(files)

data = []
for file in files:
    tmp = np.load(file)
    tmp = np.reshape(tmp, (1,41,101,1))
    tmp = upsample.predict(tmp)
    tmp = np.reshape(tmp, (82,202))
    data.append(tmp)

fig = plt.figure()
ims = []
for d in data:
    im = plt.imshow(d, animated=True)
    ims.append([im])
#Show
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('model.mp4')
