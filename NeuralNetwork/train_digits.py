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

def main():
    # Load data
    x_train, t_train = read_data('dataset/optdigits.tra')
    x_test, t_test = read_data('dataset/optdigits.tes')

    # Split data
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    index_data = np.arange(train_size)
    np.random.shuffle(index_data)
    x_train, t_train = x_train[index_data], t_train[index_data]
    x_val, t_val = x_test[:test_size // 2], t_test[:test_size // 2]
    x_test, t_test = x_test[test_size // 2:], t_test[test_size // 2:]
    t_train, t_val, t_test = np.int_(t_train), np.int_(t_val), np.int_(t_test)

    # Data normalize X ~ N(0, 1)
    x_train, _, _ = data2norm(x_train)
    x_val, _, _ = data2norm(x_val)
    x_test, _, _ = data2norm(x_test)

    # Layers net
    layers = [
        FullyConnected(64, 10),
        ReLU(),
        FullyConnected(10, 10),
        Softmax()
    ]

    # Build model and parametrs training
    model = NeuralNetwork(layers)
    criterion = Criterion('CrossEntropy')
    learning_rate = 0.0001
    reg = 0.000005
    model.set_optimizer(SGD(learning_rate, reg))
    epoches = 100

    # training
    train_error = []
    train_accuracy = []
    val_error = []
    val_accuracy = []
    for epoch in range(epoches):
        # Training
        model.zero_grad()
        output = model.forward(x_train)
        model.backward(output, t_train)
        model.step()

        output = model.forward(x_train)
        train_error.append(criterion.loss(output, t_train))
        train_accuracy.append(Accuracy(output, t_train))

        # Validation
        output = model.forward(x_val)
        val_error.append(criterion.loss(output, t_val))
        val_accuracy.append(Accuracy(output, t_val))

        print('Epoch - {} | Train loss - {}, Train accuracy - {}, Val loss - {}, Val accuracy - {}'.
              format(epoch + 1, train_error[-1], train_accuracy[-1], val_error[-1], val_accuracy[-1]))

    # Plot
    fig, axes = plt.subplots(1, 2, num='training', figsize=(12, 6))
    axes[0].plot(train_error, label='Train loss')
    axes[1].plot(train_accuracy, label='Train accuracy')
    axes[0].plot(val_error, label='Val loss')
    axes[1].plot(val_accuracy, label='Val accuracy')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    print('Test accuracy - {}'.format(Accuracy(model.forward(x_test), t_test)))
    model.save_weights()

if __name__ == '__main__':
    main()
