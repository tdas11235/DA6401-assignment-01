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


def identity(x):
    return x


def identity_backprop(grad_prev, x):
    return grad_prev


def softmax(h):
    h -= np.max(h, axis=1, keepdims=True)
    exph = np.exp(h)
    probs = exph / np.sum(exph, axis=1, keepdims=True)
    return probs


class Linear:
    def __init__(self, in_neuron, out_neuron, init_method="random", init_b=False, seed=42):
        np.random.seed(seed)
        if init_method == "Xavier":
            self.w = np.random.randn(in_neuron, out_neuron) * np.sqrt(1 / in_neuron)
            if init_b:
                self.b = np.random.randn(out_neuron) * np.sqrt(1 / in_neuron)
            else:
                self.b = np.zeros(out_neuron)
        elif init_method == "random":
            self.w = np.random.randn(in_neuron, out_neuron)
            if init_b:
                self.b = np.random.randn(out_neuron)
            else:
                self.b = np.zeros(out_neuron)
        else:
            raise NotImplementedError
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, h):
        self.h = h
        return h @ self.w + self.b.reshape(1, -1)

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

    def step(self):
        if self.optimiser:
            self.optimiser.update(self)

    def set_optimiser(self, optimiser):
        self.optimiser = optimiser
