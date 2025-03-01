import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical


def load_fashion_data():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Flatten x_train (convert from 28x28 to 1D array of 784 features)
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    # One-hot encode y_train and y_test
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)
