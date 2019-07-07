import numpy as np
from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([0,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([1,1]))

output = np.array([0,0,0,1])

percept = Perceptron(2)
percept.train(training_inputs, output)

inputs = np.array([1,1])
print(percept.predict(inputs))

inputs = np.array([1,1])
print(percept.predict(inputs))