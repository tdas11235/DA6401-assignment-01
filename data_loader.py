import numpy as np
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical


def load_fashion_data():
    # Load dataset
    (x_train_og, y_train_og), (x_test_og, y_test_og) = fashion_mnist.load_data()

    # Flatten x_train (convert from 28x28 to 1D array of 784 features)
    x_train = x_train_og.reshape(
        x_train_og.shape[0], -1).astype('float32') / 255.0
    x_test = x_test_og.reshape(
        x_test_og.shape[0], -1).astype('float32') / 255.0

    # One-hot encode y_train and y_test
    y_train = to_categorical(y_train_og, num_classes=10)
    y_test = to_categorical(y_test_og, num_classes=10)

    return (x_train, y_train), (x_test, y_test), (x_test_og, y_test_og)


def load_mnist_data():
    # Load MNIST dataset
    (x_train_og, y_train_og), (x_test_og, y_test_og) = mnist.load_data()

    # Flatten x_train (convert from 28x28 to 1D array of 784 features)
    x_train = x_train_og.reshape(
        x_train_og.shape[0], -1).astype('float32') / 255.0
    x_test = x_test_og.reshape(
        x_test_og.shape[0], -1).astype('float32') / 255.0

    # One-hot encode y_train and y_test
    y_train = to_categorical(y_train_og, num_classes=10)
    y_test = to_categorical(y_test_og, num_classes=10)

    return (x_train, y_train), (x_test, y_test), (x_test_og, y_test_og)
