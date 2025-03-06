from neural_net import Linear
import numpy as np


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
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    self.u_w[i] = self.beta * self.u_w[i] + self.lr * layer.dw
                    self.u_b[i] = self.beta * self.u_b[i] + self.lr * layer.db
                    layer.w -= self.u_w[i]
                    layer.b -= self.u_b[i]


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
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    layer.w -= self.beta * self.u_w[i]
                    layer.b -= self.beta * self.u_b[i]

    def update(self, model):
        if not self.started:
            self.start(model)
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    layer.w += self.beta * self.u_w[i]
                    layer.b += self.beta * self.u_b[i]
                    self.u_w[i] = self.beta * self.u_w[i] + self.lr * layer.dw
                    self.u_b[i] = self.beta * self.u_b[i] + self.lr * layer.db
                    layer.w -= self.u_w[i]
                    layer.b -= self.u_b[i]


class RMSProp(Optimiser):
    def __init__(self, lr, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.v_w = []
        self.v_b = []
        self.started = False

    def start(self, model):
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.v_w.append(np.zeros_like(layer.w))
                    self.v_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def update(self, model):
        if not self.started:
            self.start(model)
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    self.v_w[i] = self.beta * self.v_w[i] + \
                        (1 - self.beta) * (layer.dw ** 2)
                    self.v_b[i] = self.beta * self.v_b[i] + \
                        (1 - self.beta) * (layer.db ** 2)
                    layer.w -= self.lr * layer.dw / \
                        (np.sqrt(self.v_w[i]) + self.epsilon)
                    layer.b -= self.lr * layer.db / \
                        (np.sqrt(self.v_b[i]) + self.epsilon)
