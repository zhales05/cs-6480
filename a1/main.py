from os import write
import numpy as np

def perceptron(inputs, weights, bias):
    total = np.dot(inputs, weights)
    return step(total + bias)

def step(input):
    if input > 0:
        return 1
    return 0;

def relu(input):
    if input > 0:
        return input
    return 0;

def and_function(inputs):
    return perceptron(inputs, [1,1], -1.5)   

def or_function(inputs):
    return perceptron(inputs, [1,1], -.05)

def nand_function(inputs):
    return perceptron(inputs, [-1,-1], 1.5)

def test(func, expected):
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    
    for inp, exp in zip(inputs, expected):
        result = func(inp)
        status = 'true' if result == exp else 'false'
        print(f"input: {inp} result: {result} expected: {exp} {status}")

#test(and_function, [0,0,0,1])
#test(or_function, [0,1,1,1])
test(nand_function, [1,1,1,0])

