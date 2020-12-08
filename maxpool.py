import numpy as np
import cupy as cp
from layer import Layer

class MaxPool2d(Layer):
    def __init__(self, input_shape, pool_size):
        self.input_shape = input_shape[0:2]
        self.channels = input_shape[2]
        self.pool_size = pool_size

    def __call__(self, input):
        return maxpool_nhwc(input, self.pool_size)[0]
    
    def backward(self, grad_output):
        h, w = self.input_shape
        self.grad = maxpool_backward_nhwc(self.input_mask, grad_output)[:,0:h,0:w,:]

    def forward(self, input):
        self.output, self.input_mask = maxpool_nhwc(input, self.pool_size)

def maxpool_nhwc(input, pool_size):
    xp = cp.get_array_module(input)
    _, h, w, _ = input.shape
    pad_h, pad_w = (-h)%pool_size, (-w)%pool_size
    padded = xp.pad(input, ((0,0),(0,pad_h),(0,pad_w),(0,0)))
    return maxpool_nhwc_helper(padded, pool_size)

def maxpool_nhwc_helper(input, pool_size):
    """
    May assume h, w divisible by pool_size
    """
    xp = cp.get_array_module(input)
    n, h, w, c = input.shape
    ns, hs, ws, cs = input.strides

    windowed = xp.lib.stride_tricks.as_strided(input, (n, h//pool_size, 
            w//pool_size, pool_size,pool_size, c), (ns, hs*pool_size, 
            ws*pool_size, hs, ws, cs))

    out = windowed.max(axis=3).max(axis=3)
    maxes = (out[:,:,:,None,None,:] == windowed)
    return (out, maxes)

def maxpool_backward_nhwc(maxes, grad_out):
    n, h, w, k, l, c = maxes.shape
    m = (maxes * grad_out[:,:,:,None,None,:]).transpose(0,1,3,2,4,5)
    return m.reshape(n, h*k, w*l, c)