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
    def __init__(self, in_neuron, out_neuron, init_method="random"):
        if init_method == "xavier":
            self.w = np.random.randn(in_neuron, out_neuron) * (1 / in_neuron)
            # self.b = np.random.randn(out_neuron) * (1 / in_neuron)
        else:
            self.w = np.random.randn(in_neuron, out_neuron)
            # self.b = np.random.randn(out_neuron)
        self.b = np.zeros(out_neuron)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, h):
        self.h = h
        return h @ self.w + self.b.reshape(1, -1)

    def backprop(self, grad_prev):
        self.dw = self.h.T @ grad_prev
        self.db = np.sum(grad_prev, axis=0)
        print("Max weight gradient:", np.max(self.dw))
        print("Max bias gradient:", np.max(self.db))
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


class Optimiser:
    def __init__(self, lr):
        self.lr = lr

    def update(self, model):
        pass


class GD(Optimiser):
    def __init__(self, lr):
        super().__init__(lr)

    def update(self, model):
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.w -= self.lr * layer.dw
                    layer.b -= self.lr * layer.db


class MGD(Optimiser):
    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.u_w = []
        self.u_b = []
        self.started = False

    def start(self, model):
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.u_w.append(np.zeros_like(layer.w))
                    self.u_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def update(self, model):
        if not self.started:
            self.start(model)
        if hasattr(model, "layers") and isinstance(model.layers, list):
            i = 0
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.u_w[i] = self.beta * self.u_w[i] + layer.dw
                    self.u_b[i] = self.beta * self.u_b[i] + layer.db
                    layer.w -= self.lr * self.u_w[i]
                    layer.b -= self.lr * self.u_b[i]
                    i += 1


class NAG(Optimiser):
    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.u_w = []
        self.u_b = []
        self.started = False

    def start(self, model):
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.u_w.append(np.zeros_like(layer.w))
                    self.u_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def lookahead(self, model):
        if not self.started:
            self.start(model)
        if hasattr(model, "layers") and isinstance(model.layers, list):
            i = 0
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.w -= self.beta * self.u_w[i]
                    layer.b -= self.beta * self.u_b[i]
                    i += 1

    def update(self, model):
        if not self.started:
            self.start(model)
        if hasattr(model, "layers") and isinstance(model.layers, list):
            i = 0
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.w += self.beta * self.u_w[i]
                    layer.b += self.beta * self.u_b[i]
                    self.u_w[i] = self.beta * self.u_w[i] + layer.dw
                    self.u_b[i] = self.beta * self.u_b[i] + layer.db
                    layer.w -= self.lr * self.u_w[i]
                    layer.b -= self.lr * self.u_b[i]
                    i += 1
