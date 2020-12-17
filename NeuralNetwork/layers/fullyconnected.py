import numpy as np
from layers.param import *

# Class parametr with value and his gradient
class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

# Fully-connected
class FullyConnected:
    def __init__(self, n_input, n_output):
        self.n_size = (n_input, n_output)
        self.W = Param(np.random.randn(n_output, n_input) * 0.1)
        self.B = Param(np.random.randn(1, n_output) * 0.1)

        self.Z_before = None

    def __str__(self):
        return "FullyConnected(n_input={}, n_output={})".format(*self.n_size)

    def params(self):
        return {'weight': self.W.value, 'bias': self.B.value}

    def zero_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    def forward(self, X, training):
        self.Z_before = X.copy()
        return X @ self.W.value.T + self.B.value

    def backward(self, d_out):
        n = self.Z_before.shape[0]
        self.W.grad += (d_out.T @ self.Z_before) / n
        self.B.grad += np.sum(d_out, axis=0, keepdims=True) / n
        return d_out @ self.W.value