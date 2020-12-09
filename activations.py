import numpy as np


def sigmoid(inputs):
    return 1 / (1 + np.exp(-1 * inputs, dtype='float64'))


def sigmoid_grad(outputs):
    return outputs * (1 - outputs)


def tan_hyperbolic(inputs):
    exponents = np.exp(inputs, dtype='float64')
    exponents_inv = np.exp(-1 * inputs, dtype='float64')
    return (exponents - exponents_inv) / (exponents + exponents_inv)


def tan_hyperbolic_grad(outputs):
    return (1 - outputs) * (1 + outputs)


def softmax(inputs):
    # inputs -= np.max(inputs)
    exponents = np.exp(inputs, dtype='float64')
    exponents_sum = np.sum(exponents)
    return exponents / exponents_sum


def softmax_grad(outputs):
    return outputs * (1 - outputs)


def log_softmax(inputs):
    exponents = np.exp(inputs, dtype='float64')
    exponents_sum = np.sum(exponents)
    return -1 * np.log(exponents / exponents_sum)


def log_softmax_grad(outputs):
    return outputs - 1


def relu(inputs):
    return np.maximum(0, inputs)


def relu_grad(outputs):
    return 1 * (outputs > 0)
