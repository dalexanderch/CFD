import numpy as np
import re
import glob
import os
import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


@threadsafe_generator
def image_generator(start, n, batch_size = 32):
    i = start
    while True:
          # Small
          path = os.getcwd() + "/small"
          files = [f for f in glob.glob(path + "**/*.npy")]
          files = sorted_nicely(files)
          x_paths  = files[i:i+batch_size]
          # Big
          path = os.getcwd() + "/big"
          files = [f for f in glob.glob(path + "**/*.npy")]
          files = sorted_nicely(files)   
          y_paths  = files[i:i+batch_size]
          
          i+=batch_size
          i%=n
          batch_input  = []
          batch_output = [] 
          
          for file in x_paths:
              data = np.load(file)
              batch_input+=[data]
          for file in y_paths:
              data = np.load(file)
              batch_output+=[data]

          # Return a tuple of (input, output) to feed the network
          batch_x = np.array(batch_input)
          batch_y = np.array(batch_output)
          batch_x = np.expand_dims(batch_x, axis=3)
          batch_y = np.expand_dims(batch_y, axis=3)
        
          yield(batch_x, batch_y)