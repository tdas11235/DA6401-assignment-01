from deepnn import *
import optimisers as opt
from tqdm import tqdm
from utils import *


def normal_trainer(X, y, in_dim, out_dim, hidden_dims, lr=0.001, optimiser_type='NAG', beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='ce', epochs=100):
    model = DNN(in_dim, out_dim, hidden_dims, act_type)
    if loss_type == 'ce':
        loss = CrossEntropy()
    else:
        loss = MSE()
    if optimiser_type == "GD":
        optimiser = opt.GD(lr)
    elif optimiser_type == "MGD":
        optimiser = opt.MGD(lr, beta)
    elif optimiser_type == "NAG":
        optimiser = opt.NAG(lr, beta)
    elif optimiser_type == "RMSProp":
        optimiser = opt.RMSProp(lr, beta)
    elif optimiser_type == "Adam":
        optimiser = opt.Adam(lr, beta1, beta2)
    elif optimiser_type == "Nadam":
        optimiser = opt.Nadam(lr, beta1, beta2)
    else:
        raise NotImplementedError
    model.set_optimiser(optimiser)
    for epoch in tqdm(range(epochs)):
        y_pred = model.forward(X)
        train_loss = loss.forward(y, y_pred)
        grad_loss = loss.backprop(y, y_pred)
        model.zero_grad()
        model.backprop(grad_loss)
        model.step()
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}")


def stochastic_trainer(optimiser, in_dim, out_dim, hidden_dims, lr, beta=None, act_type='sigmoid', loss_type='bce'):
    pass
