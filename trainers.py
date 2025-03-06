import neural_net as nn
from deepnn import *
import numpy as np


def ce(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


def ce_backprop(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]


def mse(y_true, y_pred):
    loss = np.mean((y_pred - y_true) ** 2)
    return loss


def mse_backprop(y_true, y_pred):
    return (2 / y_true.shape[0]) * y_pred * ((y_pred - y_true) -
                                             np.sum((y_pred - y_true) * y_pred, axis=1, keepdims=True))


def softmax(h):
    h -= np.max(h, axis=1, keepdims=True)
    exph = np.exp(h)
    probs = exph / np.sum(exph, axis=1, keepdims=True)
    return probs


def normal_trainer(X, y, in_dim, out_dim, hidden_dims, lr=0.001, optimiser_type='NAG', beta=0.9, act_type='sigmoid', loss_type='ce', epochs=100):
    model = DNN(in_dim, out_dim, hidden_dims, act_type)
    if optimiser_type == "GD":
        optimiser = nn.GD(lr)
    elif optimiser_type == "MGD":
        optimiser = nn.MGD(lr, beta)
    elif optimiser_type == "NAG":
        optimiser = nn.NAG(lr, beta)
    model.set_optimiser(optimiser)
    for epoch in range(epochs):
        h = model.forward(X)
        if loss_type == 'ce':
            y_pred = softmax(h)
            loss = ce(y, y_pred)
            grad_loss = ce_backprop(y, y_pred)
        else:
            y_pred = softmax(h)
            loss = mse(y, y_pred)
            grad_loss = mse_backprop(y, y_pred)
        model.zero_grad()
        model.backprop(grad_loss)
        model.step()
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


def stochastic_trainer(optimiser, in_dim, out_dim, hidden_dims, lr, beta=None, act_type='sigmoid', loss_type='bce'):
    pass
