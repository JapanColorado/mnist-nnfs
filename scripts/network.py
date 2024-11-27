
import random
import time
import numpy as np

from src.mnist_loader import load_data_wrapper

def sigmoid(z):
    '''The sigmoid function.'''
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    '''The derivative of the sigmoid function.'''
    return sigmoid(z)*(1-sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]] 
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        '''Return the output of the network if ``a`` is input.'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        n = len(training_data)

        for j in range(epochs):
            time1 = time.time()

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            time2 = time.time()

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}, took {time2-time1:.2f} seconds")
            else:
                print(f"Epoch {j} complete in {time2-time1:.2f} seconds")
    
    def update_mini_batch(self, mini_batch, eta):

        # Nabla_b and nabla_w are matrices that hold the partial derivatives
        # of the cost function with respect to each bias(nabla_b) or weight(nabla_w)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta * nw / len(mini_batch))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb / len(mini_batch))
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        '''Use an index approach to backpropagation to compute the gradient of the cost function.'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        zs = []
        activation = x
        activations = [x]

        # Forward Pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Last(L) layer weight & bias adjustment
        delta = (activations[-1] - y) * dsigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * dsigmoid(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(1 if x == y  else 0 for (x, y) in test_results)

train, valid, test = load_data_wrapper()
test_network = Network([784, 20, 20, 10])
test_network.SGD(train, 30, 10, 3, test_data=test)