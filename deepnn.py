import numpy as np
import neural_net as nn


class DNN(nn.Module):
    def __init__(self, in_dim, output_dim, hidden_dims):
        super().__init__()
        self.n = len(hidden_dims)
        self.layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        self.h = []
        self.a = []
        for i in range(self.n):
            x = self.layers[i].forward(x)
            self.h.append(x)
            x = nn.sigmoid(x)
            self.a.append(x)
        x = self.layers[self.n].forward(x)
        self.h.append(x)
        return x

    def backprop(self, grad_prev):
        for i in range(self.n, -1, -1):
            gx =
