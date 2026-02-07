import numpy as np

def test(func, expected):
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    
    for inp, exp in zip(inputs, expected):
        result = func(inp)
        status = 'true' if result == exp else 'false'
        print(f"input: {inp} result: {result} expected: {exp} {status}")
         
def relu(input):
    if input > 0:
        return input
    return 0

def relu_perceptron(inputs, weights, bias):
    total = np.dot(inputs, weights)
    return relu(total + bias)

def layer_1(inputs):
    return(relu_perceptron(inputs, [1, 1], 0))

def layer_1_2(inputs):
    return(relu_perceptron(inputs, [1, 1], - 1))

def layer_output(inputs):
    return(relu_perceptron(inputs, [1, - 1], 0))

def or_function(inputs):
    return layer_output([layer_1(inputs), layer_1_2(inputs)])

test(or_function, [0,1,1,1])




