from neural_net import Linear
import numpy as np


class Optimiser:
    def __init__(self, lr):
        self.lr = lr

    def update(self, model):
        pass


class GD(Optimiser):
    def __init__(self, lr, weight_decay=0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def update(self, model):
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.w -= self.lr * \
                        (layer.dw + self.weight_decay * layer.w)
                    layer.b -= self.lr * layer.db


class MGD(Optimiser):
    def __init__(self, lr, beta=0.9, weight_decay=0):
        super().__init__(lr)
        self.beta = beta
        self.u_w = []
        self.u_b = []
        self.started = False
        self.weight_decay = weight_decay

    def start(self, model):
        # initialize the optimiser with the velocities
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
                    self.u_w[i] = self.beta * self.u_w[i] + self.lr * \
                        (layer.dw + self.weight_decay * layer.w)
                    self.u_b[i] = self.beta * self.u_b[i] + self.lr * layer.db
                    layer.w -= self.u_w[i]
                    layer.b -= self.u_b[i]


class NAG(Optimiser):
    def __init__(self, lr, beta=0.9, weight_decay=0):
        super().__init__(lr)
        self.beta = beta
        self.u_w = []
        self.u_b = []
        self.started = False
        self.weight_decay = weight_decay

    def start(self, model):
        # initialize the optimiser with the velocities
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.u_w.append(np.zeros_like(layer.w))
                    self.u_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def lookahead(self, model):
        # lookahead step
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
                    self.u_w[i] = self.beta * self.u_w[i] + self.lr * \
                        (layer.dw + self.weight_decay * layer.w)
                    self.u_b[i] = self.beta * self.u_b[i] + self.lr * layer.db
                    layer.w -= self.u_w[i]
                    layer.b -= self.u_b[i]


class RMSProp(Optimiser):
    def __init__(self, lr, beta=0.9, epsilon=1e-8, weight_decay=0):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.v_w = []
        self.v_b = []
        self.started = False
        self.weight_decay = weight_decay

    def start(self, model):
        # initialize the optimiser with the histories
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
                        (1 - self.beta) * \
                        ((layer.dw + self.weight_decay * layer.w) ** 2)
                    self.v_b[i] = self.beta * self.v_b[i] + \
                        (1 - self.beta) * (layer.db ** 2)
                    layer.w -= self.lr * (layer.dw + self.weight_decay * layer.w) / \
                        (np.sqrt(self.v_w[i]) + self.epsilon)
                    layer.b -= self.lr * layer.db / \
                        (np.sqrt(self.v_b[i]) + self.epsilon)


class Adam(Optimiser):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = []
        self.m_b = []
        self.v_w = []
        self.v_b = []
        self.t = 0
        self.started = False
        self.weight_decay = weight_decay

    def start(self, model):
        # initialize the optimiser with the velocities and histories
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.m_w.append(np.zeros_like(layer.w))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_w.append(np.zeros_like(layer.w))
                    self.v_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def update(self, model):
        if not self.started:
            self.start(model)
        self.t += 1
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    self.m_w[i] = self.beta1 * self.m_w[i] + \
                        (1 - self.beta1) * (layer.dw + self.weight_decay * layer.w)
                    self.m_b[i] = self.beta1 * self.m_b[i] + \
                        (1 - self.beta1) * layer.db
                    self.v_w[i] = self.beta2 * self.v_w[i] + \
                        (1 - self.beta2) * \
                        ((layer.dw + self.weight_decay * layer.w) ** 2)
                    self.v_b[i] = self.beta2 * self.v_b[i] + \
                        (1 - self.beta2) * (layer.db ** 2)
                    m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                    m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                    v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                    v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                    layer.w -= self.lr * m_w_hat / \
                        (np.sqrt(v_w_hat) + self.epsilon)
                    layer.b -= self.lr * m_b_hat / \
                        (np.sqrt(v_b_hat) + self.epsilon)


class Nadam(Optimiser):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = []
        self.m_b = []
        self.v_w = []
        self.v_b = []
        self.t = 0
        self.started = False
        self.weight_decay = weight_decay

    def start(self, model):
        # initialize the optimiser with the velocities and histories
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for layer in model.layers:
                if isinstance(layer, Linear):
                    self.m_w.append(np.zeros_like(layer.w))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_w.append(np.zeros_like(layer.w))
                    self.v_b.append(np.zeros_like(layer.b))
            self.started = True
        else:
            print("Layers not found!")

    def update(self, model):
        if not self.started:
            self.start(model)
        self.t += 1
        if hasattr(model, "layers") and isinstance(model.layers, list):
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    self.m_w[i] = self.beta1 * self.m_w[i] + \
                        (1 - self.beta1) * (layer.dw + self.weight_decay * layer.w)
                    self.m_b[i] = self.beta1 * self.m_b[i] + \
                        (1 - self.beta1) * layer.db
                    self.v_w[i] = self.beta2 * self.v_w[i] + \
                        (1 - self.beta2) * \
                        ((layer.dw + self.weight_decay * layer.w) ** 2)
                    self.v_b[i] = self.beta2 * self.v_b[i] + \
                        (1 - self.beta2) * (layer.db ** 2)
                    m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                    m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                    v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                    v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                    m_w_hat = self.beta1 * m_w_hat + \
                        ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * \
                        (layer.dw + self.weight_decay * layer.w)
                    m_b_hat = self.beta1 * m_b_hat + \
                        ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * layer.db
                    layer.w -= self.lr * m_w_hat / \
                        (np.sqrt(v_w_hat) + self.epsilon)
                    layer.b -= self.lr * m_b_hat / \
                        (np.sqrt(v_b_hat) + self.epsilon)
