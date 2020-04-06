from keras.utils import Sequence
import numpy as np
class data(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        
        batch_input = np.array([np.expand_dims(np.load(file), axis=2) for file in batch_x])
        batch_output = np.array([np.expand_dims(np.load(file), axis=2) for file in batch_y])
#        batch_input = np.expand_dims(batch_input, axis=3)
#        batch_output = np.expand_dims(batch_output, axis=3)
        
        return batch_input, batch_output