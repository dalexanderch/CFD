from PIL import Image
import os, sys
import time

# Run only once to downscale the images and convert to grayscale once and for all
curr = os.getcwd()
path = "/data/small/small/"
pathsave = "/data/small/small/"
dirs = os.listdir( curr + path )
start = time.time()

for index, item in enumerate(dirs):
	print(index)
	im = Image.open(curr + path + item)
	im = im.convert('L')
	imResize = im.resize((40,100), Image.BILINEAR)
	imResize.save(curr + pathsave + item, 'JPEG')

end = time.time()
print("Time elapsed : {}".format(end - start))

# Run only once to downscale the images and convert to grayscale once and for all
curr = os.getcwd()
path = "/data/big/big/"
pathsave = "/data/big/big/"
dirs = os.listdir( curr + path )
start = time.time()

for index, item in enumerate(dirs):
	print(index)
	im = Image.open(curr + path + item)
	im = im.convert('L')
	imResize = im.resize((80,200), Image.BILINEAR)
	imResize.save(curr + pathsave + item, 'JPEG')

end = time.time()
print("Time elapsed : {}".format(end - start))
