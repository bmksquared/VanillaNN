import numpy as np


def mse(prediction, true_prediction):
    diff = prediction - true_prediction
    return (diff.dot(diff)) / len(diff)


def mse_grad(prediction, true_prediction):
    diff = prediction - true_prediction
    return (diff * 2) / len(diff)


def logistic_loss(prediction, true_prediction):
    return -1 * np.sum(true_prediction * np.log(prediction) + (1 - true_prediction) * np.log(1 - prediction))


def logistic_loss_grad(prediction, true_prediction):
    return ((1 - true_prediction) / (1 - prediction) - (true_prediction / prediction)) / len(prediction)


def cross_entropy(prediction, true_prediction):
    return -1 * np.sum(true_prediction * np.log(prediction))


def cross_entropy_grad(prediction, true_prediction):
    # return (prediction - true_prediction) / len(prediction) # Wrong but working, Why?
    return -1 * true_prediction / prediction
