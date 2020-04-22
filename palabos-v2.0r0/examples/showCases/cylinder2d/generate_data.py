import numpy as np
# import subprocess
import os
import glob
import re
import shutil
import sys
import time


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Arguments | ex : python3 ./generate_data.py 100 2 600 20 20 5 2 
# cd Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d
numBatches = int(sys.argv[1])
upRate = int(sys.argv[2]) # big = upRate * n
Re = int(sys.argv[3])
Variance = int(sys.argv[4])
Diameter = int(sys.argv[5])
lx = int(sys.argv[6])
ly = int(sys.argv[7])

# Generate random Reynolds number (sample from normal distribution with mean Re)
mu, sigma = Re, Variance       # mean and standard deviation
samples = np.random.normal(mu, sigma, numBatches)



# Generate small samples
for index,sample in enumerate(samples):
    # time
    start = time.time()

    # run ./cylinder2d
    cmd = "./cylinder2d {} {} {} {} tmp/"
    cmd = cmd.format(Re,Diameter,lx,ly)
    os.system(cmd)

    # # clean tmp
    # cmd = "cd tmp;rm *.gif"
    # os.system(cmd)

    # rename files

    path = "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/small"
    files = [f for f in glob.glob(path + "**/*.gif", recursive=True)]
    max = len(files)

    path = "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/tmp"
    files = [f for f in glob.glob(path + "**/*.gif", recursive=True)]
    n = max + 1
    for f in sorted_nicely(files):
        newpath = "{}/{}.gif".format(os.path.dirname(f),n)
        os.rename(f,newpath)
        shutil.move(newpath, "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/small/{}.gif".format(n))
        n = n + 1

    # clean tmp
    cmd = "cd tmp;rm *"
    os.system(cmd)

    # time
    stop = time.time()

    # Print end of epoch
    print("Small batch {} generated with Diameter:{} , Re:{}, lx:{}, ly:{}. Time taken : {} seconds".format(index+1,Diameter,sample,lx,ly,stop-start))


# Generate big samples
Diameter = upRate * Diameter
for index,sample in enumerate(samples):

    # time
    start = time.time()
    # run ./cylinder2d
    cmd = "./cylinder2d {} {} {} {} tmp/"
    cmd = cmd.format(Re,Diameter,lx,ly)
    os.system(cmd)

    # # convert images
    # cmd = "cd tmp;mogrify -format gif *.gif;rm *.gif"
    # os.system(cmd)

    # rename files

    path = "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/big"
    files = [f for f in glob.glob(path + "**/*.gif", recursive=True)]
    max = len(files)

    path = "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/tmp"
    files = [f for f in glob.glob(path + "**/*.gif", recursive=True)]
    n = max + 1
    for f in sorted_nicely(files):
        newpath = "{}/{}.gif".format(os.path.dirname(f),n)
        os.rename(f,newpath)
        shutil.move(newpath, "/home/david/Documents/CFD/palabos-v2.0r0/examples/showCases/cylinder2d/big/{}.gif".format(n))
        n = n + 1
    # clean tmp
    cmd = "cd tmp;rm *"
    os.system(cmd)

    # time
    stop = time.time()

    # Print end of epoch
    print("Big batch {} generated with Diameter:{} , Re:{}, lx:{}, ly:{}. Time taken : {} seconds".format(index+1,Diameter,sample,lx,ly,stop-start))
