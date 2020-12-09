import numpy as np
from Layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, num_inputs, num_outputs):
        """

        :param num_inputs: dimension of the inputs to the Layer
        :param num_outputs: dimension of outputs from the Layer
        """
        super().__init__()
        self.activation_units = num_outputs
        # size of W is num_outputs x num_inputs
        self.weights = np.random.standard_normal((self.activation_units, num_inputs)) / num_inputs
        # vector of size num_outputs
        self.biases = np.random.randn(num_outputs)

    def forward_pass(self, inputs):
        self.inputs = inputs
        # outputs is W * x + biases
        # self.outputs = np.dot(self.weights, self.inputs) + self.biases
        # return self.outputs
        return np.dot(self.weights, self.inputs) + self.biases

    def backward_pass(self, output_gradient, learning_rate=0.2):
        """

        :param output_gradient: (dL/dy) vector of size num_outputs
        :param learning_rate:
        :return:
        """
        # dL/dx = W.T * dL/dy Derive it for once
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Weights update rule is W - dL/dY * x.T
        self.weights -= learning_rate * np.outer(output_gradient, self.inputs)
        self.biases -= learning_rate * output_gradient
        return input_gradient


class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_gradient):
        super().__init__()
        self.activation_function = activation_function
        self.activation_gradient = activation_gradient

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_function(inputs)
        return self.outputs

    def backward_pass(self, output_gradient):
        """

        :param output_gradient:
        :return:
        """
        # dL/dx = dL/dy * dy/dx, y is sigmoid therefore
        activation_gradient = self.activation_gradient(self.outputs)
        input_gradient = output_gradient * activation_gradient
        return input_gradient


class LossLayer:
    def __init__(self, loss_function, loss_gradient):
        self.prediction = None
        self.desired_prediction = None
        self.loss_function = loss_function
        self.loss_gradient = loss_gradient
        self.error = 0

    def get_loss(self, prediction, desired_prediction):
        """

        :param prediction: vector of predictions y
        :param desired_prediction: actual answer to the input y*
        :return:
        """
        self.prediction = prediction
        self.desired_prediction = desired_prediction
        self.error = self.loss_function(prediction, desired_prediction)
        return self.error

    def get_gradient(self):
        """

        :return: dL/dy vector
        """
        return self.loss_gradient(self.prediction, self.desired_prediction)


class ConvolutionLayer:
    def __init__(self, input_size, filter_size, out_feature_maps, stride=1, valid_padding=False):
        """

        :param input_size: D x H x W
        :param filter_size: FH x FW
        :param out_feature_maps: Number of output feature maps (or) Number of filters to be applied
        """
        self.inputs = None
        self.outputs = None
        # Setting up all the parameters
        self.stride = stride
        self.valid_padding = valid_padding
        # input_depth is number of channels in the Input
        self.input_depth, self.input_height, self.input_width = input_size
        self.filter_height, self.filter_width = filter_size
        self.output_depth = out_feature_maps
        self.output_height, self.output_width = self.input_height, self.input_width

        self.modified_inputs = None
        self.filter_weights = np.random.standard_normal(
            (out_feature_maps, self.input_depth, self.filter_height, self.filter_width))

        if valid_padding:
            self.output_height, self.output_width = (self.input_height - self.filter_height) // stride + 1, \
                                                    (self.input_width - self.filter_width) // stride + 1
        self.biases = np.random.randn(out_feature_maps, self.output_height, self.output_width)

    def _add_padding(self, feature_maps):
        padding_height = self.filter_height // 2
        padding_width = self.filter_width // 2
        depth, height, width = feature_maps.shape
        # Assuming stride of size 1
        modified_inputs = np.zeros((depth, height + self.filter_height - 1, width + self.filter_width - 1))
        modified_inputs[:, padding_height:-padding_height, padding_width:-padding_width] = feature_maps
        return modified_inputs

    def _convolve_forward(self, inputs, filter_weights):
        outputs = np.zeros((self.output_depth, self.output_height, self.output_width))
        for out_feature_map_index in range(self.output_depth):
            for row in range(self.output_height):
                for col in range(self.output_width):
                    target_input = inputs[:, row:(row + self.filter_height), col:(col + self.filter_width)]
                    outputs[out_feature_map_index][row][col] = \
                        np.sum(filter_weights[out_feature_map_index] * target_input)
        return outputs

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.modified_inputs = inputs
        if not self.valid_padding:
            self.modified_inputs = self._add_padding(self.inputs)
        # self.outputs = self.biases + self._convolve_forward(self.modified_inputs, self.filter_weights)
        # return self.outputs
        return self._convolve_forward(self.modified_inputs, self.filter_weights) + self.biases

    def _weights_derivatives(self, modified_inputs, output_gradient):
        """

        :param modified_inputs: Cached modified(padded) inputs for back propagation
        :param output_gradient: dL/dY
        :return: dL/dW
        """
        weight_derivatives = np.zeros((self.output_depth, self.input_depth, self.filter_height, self.filter_width))
        for out_index in range(self.output_depth):
            for in_index in range(self.input_depth):
                for row in range(self.filter_height):
                    for col in range(self.filter_width):
                        target_input = \
                            modified_inputs[in_index, row:(row + self.input_height), col:(col + self.input_width)]
                        weight_derivatives[out_index][in_index][row][col] = \
                            np.sum(target_input * output_gradient[out_index])
        return weight_derivatives

    def _convolve_backward(self, output_gradient, filter_weights):
        # Flip the filter weights
        filter_weights = filter_weights[:, :, ::-1, ::-1]
        results = np.zeros((self.input_depth, self.input_height, self.input_width))
        for input_feature_map_index in range(self.input_depth):
            for row in range(self.input_height):
                for col in range(self.input_width):
                    target_output = output_gradient[:, row:(row + self.filter_height), col:(col + self.filter_width)]
                    results[input_feature_map_index][row][col] = \
                        np.sum(filter_weights[:, input_feature_map_index] * target_output)
        return results

    def backward_pass(self, output_gradient, learning_rate=0.9):
        modified_gradient = self._add_padding(output_gradient)
        # dL/dX Input Gradient
        input_gradient = self._convolve_backward(modified_gradient, self.filter_weights)
        self.biases -= learning_rate * output_gradient
        self.filter_weights -= learning_rate * self._weights_derivatives(self.modified_inputs, output_gradient)
        return input_gradient


class MaxPool:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.input_gradient_indices = None

    def forward_pass(self, inputs):
        self.inputs = inputs
        depth, height, width = inputs.shape
        self.outputs = np.zeros((inputs.shape[0], inputs.shape[1] // 2, inputs.shape[2] // 2))
        # Input Gradient Indices will be used in the Backward Pass
        self.input_gradient_indices = np.zeros_like(self.inputs)
        for depth_index in range(depth):
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    self.outputs[depth_index][row // 2][col // 2] = np.max(inputs[depth_index, row:row+2, col:col+2])
                    # Figure out the row index and the column index of the Max Element in the 2x2 Grid
                    max_index = np.argmax(inputs[depth_index, row:row+2, col:col+2])
                    row_index = max_index // 2
                    col_index = max_index % 2
                    self.input_gradient_indices[depth_index][row + row_index][col + col_index] = 1
        return self.outputs

    def backward_pass(self, output_gradient):
        depth, height, width = self.inputs.shape
        input_gradient = np.zeros_like(self.inputs)
        for depth_index in range(depth):
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    input_gradient[depth_index, row:row+2, col:col+2] = output_gradient[depth_index][row // 2][col // 2]
        input_gradient = input_gradient * self.input_gradient_indices
        return input_gradient


class Flatten:
    def __init__(self):
        self.inputs_shape = None

    def forward_pass(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.flatten()

    def backward_pass(self, output_gradient):
        return output_gradient.reshape(self.inputs_shape)
