import numpy as np
import cupy as cp
from layer import Layer
from optim import SGD

class Dense(Layer):
    def __init__(self, input_shape, output_shape, gpu=False):
        #super().__init__(input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gpu = gpu

        if gpu:
            self.weights = cp.random.random((output_shape, input_shape)).astype('float32')
            self.bias = cp.zeros(output_shape).astype('float32')
        else:
            self.weights = np.random.random((output_shape, input_shape)).astype('float32')
            self.bias = np.zeros(output_shape).astype('float32')

        self.weights_optimizer = SGD()
        self.bias_optimizer = SGD()

    def __call__(self, input):
        xp = cp.get_array_module(input)
        return xp.maximum(0, input @ self.weights.T + self.bias)

    def forward(self, input):
        super().forward(input)
    
    def backward(self, grad_output):
        grad_output *= (self.output > 0)
        grad_bias = grad_output.sum(axis=0)
        grad_weights = grad_output.T @ self.input
        self.grad = grad_output @ self.weights

        #Update internals
        self.weights += self.weights_optimizer(grad_weights)
        self.bias += self.bias_optimizer(grad_bias)