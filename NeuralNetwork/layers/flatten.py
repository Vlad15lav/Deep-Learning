import numpy as np

class Flattener:
    def __init__(self):
        self.X_shape = None

    def __str__(self):
        return "Flattener()"

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def forward(self, X, training):
        self.X_shape = X.shape
        return np.ravel(X).reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}