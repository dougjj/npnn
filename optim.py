class SGD():
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def __call__(self, delta):
        self.velocity = (self.momentum * self.velocity - 
                         self.learning_rate * delta)
        return self.velocity