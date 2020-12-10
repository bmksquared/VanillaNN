from NetworkGraph import NeuralNetwork as nG
from Layers import *
from losses import *
from activations import *
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


def main():
    # data = load_iris()
    # data = load_digits()
    # train_set, test_set, train_labels, test_labels = train_test_split(
    #     data['data'], data['target'], test_size=0.20)

    (train_set, train_labels), (test_set, test_labels) = mnist.load_data()
    train_set, test_set = train_set.reshape(60000, 784), test_set.reshape(10000, 784)

    train_max = max(train_labels)
    # test_max = max(test_labels)
    train_labels_one_hot = np.eye(train_max + 1)[train_labels]
    test_labels_one_hot = np.eye(train_max + 1)[test_labels]  # It's train_max, not a mistake
    feature_dimension = train_set.shape[1]

    neural_net = nG()
    neural_net.add_layer(FullyConnectedLayer(feature_dimension, 256))
    neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    neural_net.add_layer(FullyConnectedLayer(256, 128))
    neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    neural_net.add_layer(FullyConnectedLayer(128, 10))
    # neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    # neural_net.add_layer(LossLayer(logistic_loss, logistic_loss_grad))
    # neural_net.add_layer(LossLayer(mse, mse_grad))
    neural_net.add_layer(SoftMaxLayer())
    neural_net.add_layer(LossLayer(cross_entropy, cross_entropy_grad))

    # Train the network on Data
    neural_net.fit(train_set, train_labels_one_hot)
    # Predictions on Test
    neural_net.test(test_set, test_labels_one_hot)

    print("******* Test Results *******")
    print(f"Test accuracy is {100 * neural_net.accuracy:.2f}")
    print(f"Test loss is {neural_net.test_loss:.5f}")


if __name__ == '__main__':
    main()
