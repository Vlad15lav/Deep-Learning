import numpy as np

class SGD:
    def __init__(self, lr, reg=0, momentum=0.9):
        self._lr = lr
        self._reg = reg
        self._momentum = momentum
        self._cache_vx = {}

    def _init_cache(self, layers):
        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw, db = l.W.grad, l.B.grad
                if dw is None or db is None:
                    continue

                dw_key, db_key = SGD._get_keys(idx)

                self._cache_vx[dw_key] = np.zeros_like(dw)
                self._cache_vx[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_keys(idx):
        return f"dw{idx}", f"db{idx}"

    def update_weight(self, layers):
        if self._reg:
            for l in reversed(layers):
                if hasattr(l, 'W'):
                    l.W.grad += reg * 2 * l.W.value

        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw_key, db_key = SGD._get_keys(idx)

                self._cache_vx[dw_key] = self._momentum * self._cache_vx[dw_key] + l.W.grad
                self._cache_vx[db_key] = self._momentum * self._cache_vx[db_key] + l.B.grad

                l.W.value -= self._lr * self._cache_vx[dw_key]
                l.B.value -= self._lr * self._cache_vx[db_key]

class RMSProp:
    def __init__(self, lr, reg=0, beta=0.9, eps=1e-8):
        self._lr = lr
        self._reg = reg
        self._cache = {}
        self._beta = beta
        self._eps = eps

    def _init_cache(self, layers):
        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw, db = l.W.grad, l.B.grad
                if dw is None or db is None:
                    continue

                dw_key, db_key = RMSProp._get_keys(idx)

                self._cache[dw_key] = np.zeros_like(dw)
                self._cache[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_keys(idx):
        return f"dw{idx}", f"db{idx}"

    def update_weight(self, layers):
        if len(self._cache_s) == 0 or len(self._cache_v) == 0:
            self._init_cache(layers)

        if self._reg:
            for l in reversed(layers):
                if hasattr(l, 'W'):
                    l.W.grad += reg * 2 * l.W.value

        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw_key, db_key = RMSProp._get_keys(idx)

                self._cache[dw_key] = self._beta * self._cache[dw_key] + \
                                      (1 - self._beta) * np.square(l.W.grad)
                self._cache[db_key] = self._beta * self._cache[db_key] + \
                                      (1 - self._beta) * np.square(l.B.grad)

                dw = l.W.grad / (np.sqrt(self._cache[dw_key]) + self._eps)
                db = l.B.grad / (np.sqrt(self._cache[db_key]) + self._eps)

                l.W.value -= self._lr * dw
                l.B.value -= self._lr * db

class Adam:
    def __init__(self, lr, reg=0, beta1=0.9, beta2=0.999, eps=1e-8):
        self._lr = lr
        self._reg = reg
        self._beta1 = beta1
        self._beta2 = beta2
        self._cache_v = {}
        self._cache_s = {}
        self._eps = eps

    def _init_cache(self, layers):
        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw, db = l.W.grad, l.B.grad
                if dw is None or db is None:
                    continue

                dw_key, db_key = Adam._get_keys(idx)

                self._cache_v[dw_key] = np.zeros_like(dw)
                self._cache_v[db_key] = np.zeros_like(db)
                self._cache_s[dw_key] = np.zeros_like(dw)
                self._cache_s[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_keys(idx):
        return f"dw{idx}", f"db{idx}"

    def update_weight(self, layers):
        if len(self._cache_s) == 0 or len(self._cache_v) == 0:
            self._init_cache(layers)

        if self._reg:
            for l in reversed(layers):
                if hasattr(l, 'W'):
                    l.W.grad += reg * 2 * l.W.value

        for idx, l in enumerate(layers):
            if hasattr(l, 'W'):
                dw_key, db_key = Adam._get_keys(idx)

                self._cache_v[dw_key] = self._beta1 * self._cache_v[dw_key] + \
                                        (1 - self._beta1) * l.W.grad
                self._cache_v[db_key] = self._beta1 * self._cache_v[db_key] + \
                                        (1 - self._beta1) * l.B.grad

                self._cache_s[dw_key] = self._beta2 * self._cache_s[dw_key] + \
                                        (1 - self._beta2) * np.square(l.W.grad)
                self._cache_s[db_key] = self._beta2 * self._cache_s[db_key] + \
                                        (1 - self._beta2) * np.square(l.B.grad)

                dw = self._cache_v[dw_key] / (np.sqrt(self._cache_s[dw_key]) + self._eps)
                db = self._cache_v[db_key] / (np.sqrt(self._cache_s[db_key]) + self._eps)

                l.W.value -= self._lr * dw
                l.B.value -= self._lr * db