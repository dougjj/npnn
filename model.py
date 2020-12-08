import numpy as np
from scipy.special import softmax
from tqdm import tqdm

class Model():
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.layers = []

    def set_loss(self, loss):
        self.loss = loss

    def __call__(self, data):
        self.layers[0].forward(data)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i-1].output)
        
        probs = softmax(self.layers[self.num_layers-1].output, axis=1)
        return probs

    def train(self, data, labels, batch_size=32, epochs=1):
        n = len(data)
        for i in range(epochs):
            shuffled_indices = np.random.permutation(n)
            self.epoch(data[shuffled_indices], labels[shuffled_indices],
                       batch_size)
            
    def epoch(self, data, labels, batch_size=32):
        n = data.shape[0]
        print(n)
        for i in tqdm(range(0,n,batch_size)):
            self.backprop(data[i:i+batch_size], labels[i:i+batch_size])

    def backprop(self, data, labels):
        self.layers[0].forward(data)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i-1].output)
        
        probs = softmax(self.layers[self.num_layers-1].output, axis=1)
        
        self.layers[self.num_layers-1].backward(probs-labels)
        for i in reversed(range(1, self.num_layers)):
            self.layers[i-1].backward(self.layers[i].grad)