import numpy as np


class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backprop(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropy(Loss):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def backprop(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]


class MSE(Loss):
    def forward(self, y_true, y_pred):
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backprop(self, y_true, y_pred):
        return (2 / y_true.shape[0]) * y_pred * ((y_pred - y_true) -
                                                 np.sum((y_pred - y_true) * y_pred, axis=1, keepdims=True))
