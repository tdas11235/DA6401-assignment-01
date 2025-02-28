import neural_net as nn
from deepnn import *
import numpy as np


def normal_trainer(X, y, in_dim, out_dim, hidden_dims, lr, optimiser_type='NAG', beta=None, act_type='sigmoid', loss_type='bce', epochs=100):
    model = DNN(in_dim, out_dim, hidden_dims, act_type)
    if optimiser_type == "GD":
        optimiser = nn.GD(lr=0.01)
    elif optimiser_type == "Momentum":
        optimiser = nn.MGD(lr=0.01, beta=0.9)
    elif optimiser_type == "NAG":
        optimiser = nn.NAG(lr=0.01, beta=0.9)
    model.set_optimiser(optimiser)
    for epoch in range(epochs):
        h = model.forward(X)
        y_pred = h
        loss = np.mean((y_pred - y) ** 2)
        grad_loss = 2 * (y_pred - y) / y.shape[0]
        model.zero_grad()
        model.backward(grad_loss)
        model.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


def stochastic_trainer(optimiser, in_dim, out_dim, hidden_dims, lr, beta=None, act_type='sigmoid', loss_type='bce'):
    pass
