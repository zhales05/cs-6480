from os import write
import numpy as np

def relu_perceptron(inputs, weights, bias):
    total = np.dot(inputs, weights)
    return relu(total + bias)

def relu(input):
    if input > 0:
        return input
    return 0;

def and_function(inputs):
    return relu_perceptron(inputs, [1,1], -1)   

def nor_function(inputs):
    return relu_perceptron(inputs, [-1, -1], 1)

def xor_function(inputs):
    return nor_function([and_function(inputs), nor_function(inputs)])

def test(func, expected):
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    
    for inp, exp in zip(inputs, expected):
        result = func(inp)
        status = 'true' if result == exp else 'false'
        print(f"input: {inp} result: {result} expected: {exp} {status}")

#test(and_function, [0,0,0,1])
test(nor_function, [1,0,0,0])
#test(xor_function, [0,1,1,0])

