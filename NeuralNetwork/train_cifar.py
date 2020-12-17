from utils.tools import *
from activation.activation import *
from layers.convolution import *
from layers.dropout import *
from layers.flatten import *
from layers.fullyconnected import *
from layers.pooling import *
from net.net import *
from optimizers.optimizers import *
from criterions.criterions import *
from metrics.metrics import *

import matplotlib.pyplot as plt


def train(model, TrainLoader, ValidLoader, epochs, criterion):
    plt.figure('Training')
    plt.ion()

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        train_loss_batch = []
        val_loss_batch = []
        # Train
        for batch in TrainLoader:
            img = batch['imgs'] / 255
            targets = batch['targets']

            model.zero_grad()
            output = model.forward(img, training=True)
            model.backward(output, targets)
            model.step()

            output = model.forward(img, training=False)
            train_loss_batch.append(criterion.loss(output, targets))

        # Validation
        for batch in ValidLoader:
            img = batch['imgs'] / 255
            targets = batch['targets']

            output = model.forward(img, training=False)
            val_loss_batch.append(criterion.loss(output, targets))

        train_loss.append(np.nanmean(train_loss_batch))
        val_loss.append(np.nanmean(val_loss_batch))
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss, label='Valid loss')
        plt.legend()
        plt.pause(0.1)

    plt.ioff()

def main():
    path_batches = 'dataset/cifar-10-batches-py/'
    files = {'train': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'],
             'valid': ['data_batch_5'],
             'test': ['test_batch']}

    TrainLoader = getLoader(path=path_batches, samples=files['train'], batch_size=25, shuff=True)
    ValidLoader = getLoader(path=path_batches, samples=files['valid'], batch_size=25)
    TestLoader = getLoader(path=path_batches, samples=files['test'], batch_size=25)

    layers = [
        Conv2D(3, 32, kernel_size=(5, 5), stride=1, padding='same'),
        ReLU(),
        MaxPooling(pool_size=(2, 2), stride=2),
        Conv2D(32, 32, kernel_size=(5, 5), stride=1, padding='same'),
        ReLU(),
        MaxPooling(pool_size=(2, 2), stride=2),
        Conv2D(32, 64, kernel_size=(5, 5), stride=1, padding='same'),
        ReLU(),
        MaxPooling(pool_size=(2, 2), stride=2),
        Flattener(),
        FullyConnected(1024, 10),
        Softmax()
    ]

    model = NeuralNetwork(layers)
    model.set_optimizer(Adam(0.001))
    criterion = Criterion('CrossEntropy')
    train(model, TrainLoader, ValidLoader, 50, criterion)

    print('Test Accuracy - {}'.format(batchAccuracy(model, TestLoader)))

if __name__ == '__main__':
    main()
