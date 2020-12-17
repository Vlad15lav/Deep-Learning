import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data(path_data):
    with open(path_data, 'r') as file:
        txt = [x.split(',') for x in file.read().splitlines()]
    data = np.array(txt, dtype=np.float32)
    return data[:, :-1], data[:, -1]

def getLoader(path, samples, batch_size, shuff=False):
    load = []
    for f in samples:
        data = unpickle(path + f)
        x_set = data[b'data']
        t_set = np.array(data[b'labels'])

        # Shuffle
        if shuff:
            index = np.arange(x_set.shape[0])
            np.random.shuffle(index)
            x_set = x_set[index]
            t_set = t_set[index]

        x_set = x_set.reshape(x_set.shape[0], 3, 32, 32)
        x_set = x_set.transpose(0, 2, 3, 1).astype('uint8')

        for i in range(x_set.shape[0] // batch_size):
            load.append({'imgs': x_set[batch_size * i:batch_size * (i + 1)],
                         'targets': t_set[batch_size * i:batch_size * (i + 1)]})
    return load

def data2norm(x_set):
    x = np.copy(x_set)
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x -= mu
    sigma[sigma == 0] = 1
    x /= sigma
    return x, mu, sigma