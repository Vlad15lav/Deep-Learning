import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layers):
        self.__layers = layers
        self.__optimizer = None

    def __getitem__(self, id):
        return self.__layers[id]

    def __str__(self):
        str_net = "NeuralNetwork {\n"
        for l in self.__layers:
            str_net += "\t{}\n".format(str(l))
        str_net += "}"
        return str_net

    def save_weights(self, name='weights'):
        f = open(r'{}.pickle'.format(name), 'wb')
        obj = self.params()
        pickle.dump(obj, f)
        print('The model is saved to file - {}.pickle!'.format(name))
        f.close()

    def load_weights(self, weights=None, path='weights.pickle'):
        if weights is None:
            f = open(path, 'rb')
            obj = pickle.load(f)
            f.close()
        else:
            obj = weights
        i = 0
        for l in self.__layers:
            if not hasattr(l, 'W'):
                continue
            l.W.value = obj[i]['weight']
            l.B.value = obj[i]['bias']
            i += 1

    def params(self):
        return [l.params() for l in self.__layers if hasattr(l, 'W')]

    def initNorm(self, sigma):
        for l in self.__layers:
            if hasattr(l, 'W'):
                l.W.value = np.random.randn(*l.W.value.shape) * sigma
                l.B.value = np.random.randn(*l.B.value.shape) * sigma

    def initHe(self):
        for l in self.__layers:
            if hasattr(l, 'W'):
                l.W.value = np.random.randn(*l.W.value.shape) * np.sqrt(2 / l.n_size[0])
                l.B.value = np.random.randn(*l.B.value.shape) * np.sqrt(2 / l.n_size[0])

    def initXavier(self):
        for l in self.__layers:
            if hasattr(l, 'W'):
                l.W.value = np.random.randn(*l.W.value.shape) * np.sqrt(2 / (l.n_size[0] + l.n_size[1]))
                l.B.value = np.random.randn(*l.B.value.shape) * np.sqrt(2 / (l.n_size[0] + l.n_size[1]))

    def zero_grad(self):
        for l in self.__layers:
            l.zero_grad()

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def forward(self, x_set, training):
        x_ = np.copy(x_set)
        for l in self.__layers:
            x_ = l.forward(x_, training)
        return x_

    def backward(self, net_out, t_set):
        d_out = net_out.copy()
        t_mask = np.zeros(d_out.shape)
        t_mask[np.arange(len(t_set)), t_set.reshape(1, -1)] = 1
        d_out -= t_mask
        for l in reversed(self.__layers):
            d_out = l.backward(d_out)

    def step(self):
        self.__optimizer.update_weight(self.__layers)