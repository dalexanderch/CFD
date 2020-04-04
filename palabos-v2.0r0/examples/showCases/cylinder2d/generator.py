import numpy as np
import re
import glob
import os

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def read(file, width, height):
    data = np.loadtxt(file)
    data = np.resize(data, (width, height))
    data = data.transpose()
    return data

def image_generator(batch_size = 32):
    i = 0
    while True:
          # Small
          path = os.getcwd() + "/small"
          files = [f for f in glob.glob(path + "**/*.dat", recursive=True)]
          files = sorted_nicely(files)
          x_paths  = files[i:i+batch_size]
          # Big
          path = os.getcwd() + "/big"
          files = [f for f in glob.glob(path + "**/*.dat", recursive=True)]
          files = sorted_nicely(files)   
          y_paths  = files[i:i+batch_size]
          
          i+=batch_size
          batch_input  = []
          batch_output = [] 
          
          for file in x_paths:
              data = read(file, 101, 41)
              batch_input+=[data]
          for file in y_paths:
              data = read(file, 202, 82)
              batch_output+=[data]

          # Return a tuple of (input, output) to feed the network
          batch_x = np.array(batch_input)
          batch_y = np.array(batch_output)
          batch_x = np.expand_dims(batch_x, axis=3)
          batch_y = np.expand_dims(batch_y, axis=3)
        
          yield(batch_x, batch_y)
          
          

