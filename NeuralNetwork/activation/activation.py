import numpy as np

# Activation function
class ReLU:
    def __init__(self):
        self.deriv = None

    def __str__(self):
        return "ReLU()"

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def forward(self, X, training):
        self.deriv = (X > 0)
        return X * self.deriv

    def backward(self, d_out):
        d_result = d_out * self.deriv
        return d_result


class Sigmoid:
    def __init__(self):
        self.deriv = None

    def __str__(self):
        return "Sigmoid()"

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def forward(self, X, training):
        self.deriv = (1 / (1 + np.exp(-X))) * (1 - 1 / (1 + np.exp(-X)))
        return 1 / (1 + np.exp(-X))

    def backward(self, d_out):
        d_result = d_out * self.deriv
        return d_result


class Softmax:
    def __init__(self):
        self.deriv = None

    def __str__(self):
        return "Softmax()"

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def forward(self, Z_last, training):
        Z_last -= np.max(Z_last, axis=1).T.reshape((Z_last.shape[0], 1))
        return np.exp(Z_last) / np.sum(np.exp(Z_last), axis=1).T.reshape((Z_last.shape[0], 1))

    def backward(self, d_out):
        return d_out
