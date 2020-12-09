import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, new_layer):
        """

        :param new_layer: New Layer to be added to the Network
        :return:
        """
        self.layers.append(new_layer)

    def fit(self, training_data, training_labels, num_epochs=100):
        # Stochastic Gradient
        train_errors = []
        for epoch in range(num_epochs):
            train_error_total = 0
            for sample_index, train_sample in enumerate(training_data):
                # print(f"Sample index {sample_index}")
                layer_input = self.predict(train_sample)
                # Loss Layer
                loss_layer = self.layers[-1]
                # Compute the loss
                train_sample_error = loss_layer.get_loss(layer_input, training_labels[sample_index])
                train_error_total += train_sample_error
                train_errors.append(train_sample_error)
                # Backward Pass the gradient
                layer_gradient = loss_layer.get_gradient()
                # layer_index = len(self.layers) - 2
                for layer in self.layers[-2::-1]:
                    # print(f"Backpropagation through layer{layer_index}")
                    layer_gradient = layer.backward_pass(layer_gradient)
                    # layer_index -= 1
            if not epoch % 10:
                print(f"Loss after epoch {epoch} is {train_error_total:.5f}")

    def predict(self, sample):
        # Forward Pass the data
        layer_input = sample
        for layer in self.layers[:-1]:
            layer_input = layer.forward_pass(layer_input)

        return layer_input

    def test(self, test_data, test_labels):
        test_error_total = 0
        # test_errors = []
        for sample_index, test_sample in enumerate(test_data):
            layer_input = self.predict(test_sample)
            # Loss Layer
            loss_layer = self.layers[-1]
            # Compute the loss
            test_sample_error = loss_layer.get_loss(layer_input, test_labels[sample_index])
            test_error_total += test_sample_error
            # test_errors.append(test_sample_error)

    def get_accuracy(self, test_data, test_labels):
        correct_samples = 0
        for sample_index, test_sample in enumerate(test_data):
            layer_input = self.predict(test_sample)
            if np.argmax(layer_input) == test_labels[sample_index]:
                correct_samples += 1

        return correct_samples / len(test_labels)
