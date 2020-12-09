from Layers import *
from losses import *
from activations import *
from NetworkGraph import NeuralNetwork as netGraph
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_map = pickle.load(fo, encoding='bytes')
    return data_map


def main():
    file_path = "test_batch"
    img_data_binary = unpickle(file_path)
    img_data = {}
    for key, value in img_data_binary.items():
        img_data[key.decode()] = value

    # print(img_data.keys())
    # for key in img_data.keys():
    #     print(len(img_data[key]))

    images, image_labels = img_data['data'][:500], img_data['labels'][:500]
    images = images.reshape(500, 3, 32, 32)
    # print(images.shape)

    train_set, test_set, train_labels, test_labels = train_test_split(
        images, image_labels, test_size=0.20)

    train_max = max(train_labels)
    test_max = max(test_labels)
    train_labels_one_hot = np.eye(train_max + 1)[train_labels]
    test_labels_one_hot = np.eye(test_max + 1)[test_labels]
    # feature_dimension = train_set.shape[1]

    conv_neural_net = netGraph()
    conv_neural_net.add_layer(ConvolutionLayer((3, 32, 32), (3, 3), 64))
    conv_neural_net.add_layer(ActivationLayer(relu, relu_grad))
    conv_neural_net.add_layer(MaxPool())

    conv_neural_net.add_layer(ConvolutionLayer((64, 16, 16), (3, 3), 128))
    conv_neural_net.add_layer(ActivationLayer(relu, relu_grad))
    conv_neural_net.add_layer(MaxPool())

    conv_neural_net.add_layer(ConvolutionLayer((128, 8, 8), (3, 3), 256))
    conv_neural_net.add_layer(ActivationLayer(relu, relu_grad))
    conv_neural_net.add_layer(MaxPool())

    conv_neural_net.add_layer(ConvolutionLayer((256, 4, 4), (3, 3), 512))
    conv_neural_net.add_layer(ActivationLayer(relu, relu_grad))
    conv_neural_net.add_layer(MaxPool())

    conv_neural_net.add_layer(Flatten())

    conv_neural_net.add_layer(FullyConnectedLayer(2 ** 11, 128))
    conv_neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))

    conv_neural_net.add_layer(FullyConnectedLayer(128, 256))
    conv_neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))

    conv_neural_net.add_layer(FullyConnectedLayer(256, 512))
    conv_neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))

    conv_neural_net.add_layer(FullyConnectedLayer(512, 1024))
    conv_neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))

    conv_neural_net.add_layer(FullyConnectedLayer(1024, 10))

    # conv_neural_net.add_layer(ActivationLayer(sigmoid, sigmoid_grad))
    # conv_neural_net.add_layer(LossLayer(logistic_loss, logistic_loss_grad))

    conv_neural_net.add_layer(SoftMaxLayer())
    conv_neural_net.add_layer(LossLayer(cross_entropy, cross_entropy_grad))

    # Train the network on Data
    conv_neural_net.fit(train_set, train_labels_one_hot)
    # Predictions on Test
    # conv_neural_net.test(test_set, test_labels_one_hot)

    print(f"Test accuracy of the model is {100 * conv_neural_net.get_accuracy(test_set, test_labels):.2f}")


if __name__ == "__main__":
    main()
