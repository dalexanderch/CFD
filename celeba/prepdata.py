from PIL import Image
import os, sys

# Run only once to downscale the images once and for all
curr = os.getcwd()
path = "/data/big/data/"
pathsave = "/data/small/data/"

dirs = os.listdir( curr + path )
for index, item in enumerate(dirs):
	print (item)
    if os.path.isfile(path+item):
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        imResize = im.resize((89,109), Image.BILINEAR)
        imResize.save(pathsave + f, 'JPEG')

