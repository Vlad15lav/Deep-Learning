import numpy as np

class Dropout:
    def __init__(self, keep_prob):
        self._keep_prob = keep_prob
        self._mask = None

    def __str__(self):
        return "Dropout(keep_prob={})".format(self._keep_prob)

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def forward(self, X, training):
        if training:
            self._mask = (np.random.rand(*X.shape) < self._keep_prob)
            return self._apply_mask(X, self._mask)
        else:
            return X

    def backward(self, d_out):
        return self._apply_mask(d_out, self._mask)

    def _apply_mask(self, array, mask):
        array *= mask
        array /= self._keep_prob
        return array