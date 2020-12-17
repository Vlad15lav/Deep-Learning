import numpy as np

class MaxPooling:
    def __init__(self, pool_size, stride):
        self._pool_size = pool_size
        self._stride = stride
        self._Z_before = None
        self._cache = {}

    def __str__(self):
        return "MaxPooling(pool_size=({}, {}), stride={})" \
            .format(*self._pool_size, self._stride)

    def params(self):
        return {}

    def zero_grad(self):
        pass

    def _save_mask(self, X, cords):
        mask = np.zeros_like(X)
        n, h, w, c = X.shape
        X = X.reshape(n, h * w, c)
        idx = np.argmax(X, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask

    def forward(self, X, training):
        self._Z_before = np.copy(X)
        n, h_in, w_in, c = X.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(X_slice, (i, j))
                output[:, i, j, :] = np.max(X_slice, axis=(1, 2))
        return output

    def backward(self, d_out):
        output = np.zeros_like(self._Z_before)
        _, h_out, w_out, _ = d_out.shape
        h_pool, w_pool = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += \
                    d_out[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return output

    def params(self):
        return {}