import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.accuracy = 0
        self.test_loss = 0

    def add_layer(self, new_layer):
        """

        :param new_layer: New Layer to be added to the Network
        :return:
        """
        self.layers.append(new_layer)

    def fit(self, train_data, train_labels, num_epochs=30):
        # Stochastic Gradient
        train_data_size = train_data.shape[0]
        for epoch in range(num_epochs):
            train_error_total = 0
            correct_samples = 0
            for sample_index, train_sample in enumerate(train_data):
                if not sample_index % 1000:
                    print(f"Sample index {sample_index}")
                loss_layer_input = self.predict(train_sample)
                if np.argmax(loss_layer_input) == np.argmax(train_labels[sample_index]):
                    correct_samples += 1
                # Loss Layer
                loss_layer = self.layers[-1]
                # Compute the loss
                train_sample_error = loss_layer.get_loss(loss_layer_input, train_labels[sample_index])
                train_error_total += train_sample_error
                # Backward Pass the gradient
                layer_gradient = loss_layer.get_gradient()
                # layer_index = len(self.layers) - 2
                for layer in self.layers[-2::-1]:
                    # print(f"Backpropagation through layer{layer_index}")
                    layer_gradient = layer.backward_pass(layer_gradient)
                    # layer_index -= 1
            # if not epoch % 10:
            print(f"After epoch {epoch}:")
            print(f"Train Loss: {train_error_total:.5f}, Train accuracy: {100 * correct_samples / train_data_size:.2f}")

    def predict(self, sample):
        """
        Predicts the output of the network for one sample
        :param sample: Input Sample
        :return:
        """
        # Forward Pass the data
        layer_input = sample
        for layer in self.layers[:-1]:
            layer_input = layer.forward_pass(layer_input)

        return layer_input

    def test(self, test_data, test_labels):
        """

        :param test_data: Test data for Model Evaluation
        :param test_labels: Test Labels One hot encoded
        :return:
        """
        test_data_size = test_labels.shape[0]
        test_error_total = 0
        correct_samples = 0
        for sample_index, test_sample in enumerate(test_data):
            loss_layer_input = self.predict(test_sample)
            if np.argmax(loss_layer_input) == np.argmax(test_labels[sample_index]):
                correct_samples += 1
            # Loss Layer
            loss_layer = self.layers[-1]
            # Compute the loss
            test_sample_error = loss_layer.get_loss(loss_layer_input, test_labels[sample_index])
            test_error_total += test_sample_error

        self.accuracy = correct_samples / test_data_size
        self.test_loss = test_error_total

    def get_accuracy(self, test_data, test_labels):
        """
        This method is not required

        :param test_data: Test Data
        :param test_labels: Test labels not One hot encoded
        :return:
        """
        correct_samples = 0
        for sample_index, test_sample in enumerate(test_data):
            layer_input = self.predict(test_sample)
            if np.argmax(layer_input) == test_labels[sample_index]:
                correct_samples += 1

        return correct_samples / len(test_labels)
