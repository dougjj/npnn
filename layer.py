from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, input):
        pass

    def forward(self, input):
        self.input = input
        self.output = self(input)

    @abstractmethod
    def backward(self, grad_output):
        pass