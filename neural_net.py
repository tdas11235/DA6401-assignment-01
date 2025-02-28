import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backprop(grad_prev, x):
    a = sigmoid(x)
    return grad_prev * a * (1 - a)


def relu(x):
    return np.maximum(x, 0)


def relu_backprop(grad_prev, x):
    grad = grad_prev.copy()
    grad[x <= 0] = 0
    return grad


def tanh(x):
    return np.tanh(x)


def tanh_backprop(grad_prev, x):
    a = tanh(x)
    return grad_prev * (1 - a ** 2)


class Linear:
    def __init__(self, in_neuron, out_neuron):
        self.w = np.random.randn(in_neuron, out_neuron)
        self.b = np.random.randn(out_neuron)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, h):
        self.h = h
        return h @ self.w + self.b

    def backprop(self, grad_prev):
        self.dw = self.h.T @ grad_prev
        self.db = np.sum(grad_prev, axis=0)
        return grad_prev @ self.w.T


class Module:
    def __init__(self):
        assert isinstance(
            self, Module), "Module.__init__() not called by its subclass"

    def parameters(self):
        params = []
        if hasattr(self, "layers") and isinstance(self.layers, list):
            for layer in self.layers:
                if isinstance(layer, Linear):
                    params.append(layer.w)
                    params.append(layer.b)
        return params

    def zero_grad(self):
        if hasattr(self, "layers") and isinstance(self.layers, list):
            for layer in self.layers:
                if isinstance(layer, Linear):
                    layer.dw.fill(0)
                    layer.db.fill(0)
