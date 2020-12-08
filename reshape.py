from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, input):
        n = input.shape[0]
        return input.reshape((n,)+self.output_shape)

    def forward(self, input):
        super().forward(input)

    def backward(self, grad_output):
        n = len(grad_output)
        self.grad = grad_output.reshape((n,)+self.input_shape)