from layers.fastconv import im2col, col2im
from layers.param import *
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding='valid'):
        self.filter_size = kernel_size
        self.n_size = (in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(kernel_size[0], kernel_size[1],
                            in_channels, out_channels) * 0.1
        )

        self.B = Param(np.random.randn(out_channels) * 0.1)

        self._padding = padding
        self._stride = stride
        self._Z_before = None
        self._cols = None

    def __str__(self):
        return "Conv2D(in_channels={}, out_channels={}, kernel_size=({}, {}), stride={}, padding={})" \
            .format(*self.n_size, *self.n_size, self._stride, self._padding)

    def params(self):
        return {'W': self.W.value, 'B': self.B.value}

    def zero_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    def __calculate_output_dims(self, input_dims):
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self.W.value.shape
        if self._padding == 'same':
            return n, h_in, w_in, n_f
        elif self._padding == 'valid':
            h_out = (h_in - h_f) // self._stride + 1
            w_out = (w_in - w_f) // self._stride + 1
            return n, h_out, w_out, n_f
        else:
            raise Exception("Invalid padding!")

    def __calculate_pad_dims(self):
        if self._padding == 'same':  # ZeroPadding
            h_f, w_f, _, _ = self.W.value.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self._padding == 'valid':
            return 0, 0
        else:
            raise Exception("Invalid padding!")

    def forward(self, X, training):
        self._Z_before = np.copy(X)

        n, h_out, w_out, _ = self.__calculate_output_dims(input_dims=X.shape)
        h_f, w_f, _, n_f = self.W.value.shape
        pad = self.__calculate_pad_dims()
        w = np.transpose(self.W.value, (3, 2, 0, 1))

        self._cols = im2col(
            array=np.moveaxis(X, -1, 1),
            filter_dim=(h_f, w_f),
            pad=pad[0],
            stride=self._stride
        )

        result = w.reshape((n_f, -1)).dot(self._cols)
        output = result.reshape(n_f, h_out, w_out, n)

        return output.transpose(3, 1, 2, 0) + self.B.value

    def backward(self, d_out):
        n, h_out, w_out, _ = self.__calculate_output_dims(
            input_dims=self._Z_before.shape)
        h_f, w_f, _, n_f = self.W.value.shape
        pad = self.__calculate_pad_dims()

        self.B.grad += d_out.sum(axis=(0, 1, 2)) / n
        d_out_reshaped = d_out.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(self.W.value, (3, 2, 0, 1))
        dw = d_out_reshaped.dot(self._cols.T).reshape(w.shape)
        self.W.grad += np.transpose(dw, (2, 3, 1, 0))

        output_cols = w.reshape(n_f, -1).T.dot(d_out_reshaped)

        output = col2im(
            cols=output_cols,
            array_shape=np.moveaxis(self._Z_before, -1, 1).shape,
            filter_dim=(h_f, w_f),
            pad=pad[0],
            stride=self._stride
        )
        return np.transpose(output, (0, 2, 3, 1))
