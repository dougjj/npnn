import numpy as np
import cupy as cp
from layer import Layer
from optim import SGD

class Conv2d(Layer):
    def __init__(self, input_shape, filters, kernel_size, gpu=False):
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.filters = filters
        input_channels = input_shape[2]

        self.gpu = gpu

        if gpu:
            self.weights = cp.random.random((kernel_size, kernel_size,
                               input_channels,filters)).astype('float32')
            self.bias = cp.zeros(self.filters).astype('float32')
        else:
            self.weights = np.random.random((kernel_size, kernel_size,
                               input_channels,filters)).astype('float32')
            self.bias = np.zeros(self.filters).astype('float32')

        self.weights_optimizer = SGD()
        self.bias_optimizer = SGD()

    def __call__(self, input):
        return np.maximum(0, cor_nhwc(input, self.weights) + self.bias)

    def forward(self, input):
        super().forward(input)
    
    def backward(self, grad_out):
        grad_out *= (self.output > 0)
        grad_bias = grad_out.sum(axis=(0,1,2)) # Sum or mean??
        grad_weights = cor_grad(self.input, grad_out)

        self.grad = cor_full(grad_out, self.weights.transpose(0,1,3,2))

        self.weights += self.weights_optimizer(grad_weights)
        self.bias += self.bias_optimizer(grad_bias)

def cor_nhwc(input, kernel):
    xp = cp.get_array_module(input)
    n, h, w, c = input.shape
    k, _, _, ch_out = kernel.shape
    im2row = xp.lib.stride_tricks.as_strided(input, 
        (n,h-k+1,w-k+1,k,k,c), input.strides[0:3]+
        input.strides[1:4]).reshape(-1,k*k*c)
    return (im2row @ kernel.reshape(-1,ch_out)).reshape(n, h-k+1,w-k+1, ch_out)

def cor_grad(input, grad_out):
    return cor_nhwc(input.transpose(3,1,2,0),grad_out.transpose(1,2,0,3)
    ).transpose(1,2,0,3)

def cor_full(input, kernel):
    xp = cp.get_array_module(input)
    k = kernel.shape[0]
    padded = xp.pad(input, [(0,0), (k-1,k-1), (k-1,k-1), (0,0)])
    return cor_nhwc(padded, kernel)