from NetworkGraph import NeuralNetwork as nG
from Layers import FullyConnectedLayer, ActivationLayer, LossLayer
from losses import *
from activations import *
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split


def main():
    data = load_digits()
    # data = load_iris()
    train_set, test_set, train_labels, test_labels = train_test_split(
        data['data'], data['target'], test_size=0.20)

    train_max = max(train_labels)
    test_max = max(test_labels)
    train_labels = np.eye(train_max + 1)[train_labels]
    test_labels_one_hot = np.eye(test_max + 1)[test_labels]
    feature_dimension = train_set.shape[1]

    neural_net = nG()
    neural_net.add_layer(FullyConnectedLayer(feature_dimension, 256))
    neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    neural_net.add_layer(FullyConnectedLayer(256, 128))
    neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    neural_net.add_layer(FullyConnectedLayer(128, 10))
    neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    # neural_net.add_layer(LossLayer(mse, mse_grad))
    neural_net.add_layer(ActivationLayer(softmax, softmax_grad))
    # neural_net.add_layer(LossLayer(cross_entropy, cross_entropy_grad))
    neural_net.add_layer(LossLayer(logistic_loss, logistic_loss_grad))

    # Train the network on Data
    neural_net.fit(train_set, train_labels)
    # Predictions on Test
    neural_net.test(test_set, test_labels_one_hot)

    print(f"Test accuracy of the model is {100 * neural_net.get_accuracy(test_set, test_labels):.2f}")


if __name__ == '__main__':
    main()
