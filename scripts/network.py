
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
        '''Train the network using minibatch stochastic gradient descent.'''
        total_time = 0
        for j in range(epochs):
            time1 = time.time()

            random.shuffle(training_data)

            # Split the data into groups with mini_batch_size datapoints in each
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            time2 = time.time()

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}, took {time2-time1:.2f} seconds")
            else:
                print(f"Epoch {j} complete in {time2-time1:.2f} seconds")

            total_time += time2-time1
            
        print(f"Final Accuracy: {self.evaluate(test_data) / len(test_data):.2f}")
        print(f"Total Training Time: {total_time:.2f}")
    
    def update_mini_batch(self, mini_batch, eta):
        '''Updates weights and biases using the gradients from one minibatch worth of data.'''

        # Empty weight and bias update matrices
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Compute gradient with respect to w/b and then add to respective update matrices
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Take step of size eta opposite to gradient
        self.weights = [w - (eta * nw / len(mini_batch))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb / len(mini_batch))
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        '''Compute the gradient of the cost function for the weights and biases for one data point.'''
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
        '''Return the number of correct predictions of the network.'''
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(1 if x == y  else 0 for (x, y) in test_results)

if __name__ == "__main__":
    train, valid, test = load_data_wrapper()
    mnist_predictor = Network([784, 20, 20, 10])
    mnist_predictor.SGD(train, 12, 10, 3, test_data=test)
    
    # Predict first 30 digits in validation set
    for i in range(30):
        print(f"Predicted: {np.argmax(mnist_predictor.feedforward(valid[i][0]))} Actual: {valid[i][1]}")