# %%
# Import libraries & helper functions
import random
import time
import numpy as np

from src.mnist_loader import load_data_wrapper

# %%
class Network(object): # Inherits from object
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]\
        # Makes a series of matricies with the right dimesions from layer to layer
        self.weights = [np.random.randn(x,y) for y,x in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''Return the output of the network if ``a`` is input.'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

# %%
dummy_net = Network([2, 3, 1])
dummy_net.feedforward([2,5])


# %%
