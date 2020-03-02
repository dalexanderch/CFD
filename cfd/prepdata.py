from PIL import Image
import os, sys
import time

# Run only once to downscale the images and convert to grayscale once and for all
curr = os.getcwd()
path = "/data/small/data/"
pathsave = "/data/small/data/"
dirs = os.listdir( curr + path )
start = time.time()

for index, item in enumerate(dirs):
	print(index)
	im = Image.open(curr + path + item)
	im = im.convert('L')
	imResize = im.resize((100,40), Image.BILINEAR)
	imResize.save(curr + pathsave + item, 'JPEG')

end = time.time()
print("Time elapsed : {}".format(end - start))

# Run only once to downscale the images and convert to grayscale once and for all
curr = os.getcwd()
path = "/data/big/data/"
pathsave = "/data/big/data/"
dirs = os.listdir( curr + path )
start = time.time()

for index, item in enumerate(dirs):
	print(index)
	im = Image.open(curr + path + item)
	im = im.convert('L')
	imResize = im.resize((100,40), Image.BILINEAR)
	imResize.save(curr + pathsave + item, 'JPEG')

end = time.time()
print("Time elapsed : {}".format(end - start))
