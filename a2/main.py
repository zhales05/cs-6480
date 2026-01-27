import numpy as np

def step_perceptron(inputs, weights, bias):
    total = np.dot(inputs, weights)
    return step(total + bias)

def step(input):
    if input > 0:
        return 1
    return 0;