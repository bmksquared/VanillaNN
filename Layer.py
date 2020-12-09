class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward_pass(self, inputs):
        self.inputs = inputs

    def backward_pass(self):
        raise NotImplementedError()
