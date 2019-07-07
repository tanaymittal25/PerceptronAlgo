import numpy as np

class Perceptron(object):
    
    def __init__(self, inputs, maxIter = 100, rate = 0.1):
        self.maxIter = maxIter
        self.rate = rate
        self.weights = np.zeros(inputs + 1)
    
    def predict(self, inpputs):
        output = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if output > 0:
            activation = 1
        else:
            activation = 0
        return activation
    
    def train(self, train_inputs, labels):
        for _ in range(self.maxIter):
            for inputs, label in zip(train_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.rate * (label - prediction) * inputs
                self.weights[0] += self.rate * (label - prediction)