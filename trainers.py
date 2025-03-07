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
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        y_pred = model.forward(X)
        train_loss = loss.forward(y, y_pred)
        grad_loss = loss.backprop(y, y_pred)
        model.zero_grad()
        model.backprop(grad_loss)
        model.step()
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")


def stochastic_trainer(X, y, in_dim, out_dim, hidden_dims, batch_size=1, lr=0.001, optimiser_type='NAG', beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='ce', epochs=100):
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
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        batch_bar = tqdm(range(0, num_samples, batch_size),
                         desc=f"Epoch {epoch+1}", leave=False)
        for i in batch_bar:
            batch_indices = indices[i:i + batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            y_pred = model.forward(X_batch)
            train_loss = loss.forward(y_batch, y_pred)
            grad_loss = loss.backprop(y_batch, y_pred)
            model.zero_grad()
            model.backprop(grad_loss)
            model.step()
            epoch_loss += train_loss
            batch_bar.set_postfix(loss=train_loss)
        avg_epoch_loss = epoch_loss / (num_samples // batch_size)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_epoch_loss:.4f}")
