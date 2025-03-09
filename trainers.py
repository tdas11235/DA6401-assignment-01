from deepnn import *
import optimisers as opt
from tqdm import tqdm
from losses import *


class Trainer:
    def train(self, X, y):
        raise NotImplementedError

    def eval(self, X, y):
        raise NotImplementedError


class NormalTrainer(Trainer):
    def __init__(self, in_dim, out_dim, hidden_dims, lr=0.001, optimiser_type='nag', momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='cross_entropy', epochs=100, init_method="random", weight_decay=0):
        self.model = DNN(in_dim, out_dim, hidden_dims, act_type, init_method)
        if loss_type == 'cross_entropy':
            self.loss = CrossEntropy()
        elif loss_type == "mean_squared_error":
            self.loss = MSE()
        else:
            raise NotImplementedError
        if optimiser_type == "gd":
            self.optimiser = opt.GD(lr, weight_decay=weight_decay)
        elif optimiser_type == "momentum":
            self.optimiser = opt.MGD(lr, momentum, weight_decay=weight_decay)
        elif optimiser_type == "nag":
            self.optimiser = opt.NAG(lr, momentum, weight_decay=weight_decay)
        elif optimiser_type == "rmsprop":
            self.optimiser = opt.RMSProp(lr, beta, weight_decay=weight_decay)
        elif optimiser_type == "adam":
            self.optimiser = opt.Adam(
                lr, beta1, beta2, weight_decay=weight_decay)
        elif optimiser_type == "nadam":
            self.optimiser = opt.Nadam(
                lr, beta1, beta2, weight_decay=weight_decay)
        else:
            raise NotImplementedError
        self.model.set_optimiser(self.optimiser)
        self.epochs = epochs

    def train(self, X, y):
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            y_pred = self.model.forward(X)
            train_loss = self.loss.forward(y, y_pred)
            grad_loss = self.loss.backprop(y, y_pred)
            self.model.zero_grad()
            self.model.backprop(grad_loss)
            self.model.step()
            tqdm.write(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {train_loss:.4f}")

    def eval(self, X, y):
        y_pred = self.model.forward(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy


class StochasticTrainer(Trainer):
    def __init__(self, in_dim, out_dim, hidden_dims, batch_size=1, lr=0.001, optimiser_type='nag', momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='cross_entropy', init_method="random", weight_decay=0):
        self.model = DNN(in_dim, out_dim, hidden_dims, act_type, init_method)
        if loss_type == 'cross_entropy':
            self.loss = CrossEntropy()
        elif loss_type == "mean_squared_error":
            self.loss = MSE()
        else:
            raise NotImplementedError
        if optimiser_type == "sgd":
            self.optimiser = opt.GD(lr, weight_decay=weight_decay)
        elif optimiser_type == "momentum":
            self.optimiser = opt.MGD(lr, momentum, weight_decay=weight_decay)
        elif optimiser_type == "nag":
            self.optimiser = opt.NAG(lr, momentum, weight_decay=weight_decay)
        elif optimiser_type == "rmsprop":
            self.optimiser = opt.RMSProp(lr, beta, weight_decay=weight_decay)
        elif optimiser_type == "adam":
            self.optimiser = opt.Adam(
                lr, beta1, beta2, weight_decay=weight_decay)
        elif optimiser_type == "nadam":
            self.optimiser = opt.Nadam(
                lr, beta1, beta2, weight_decay=weight_decay)
        else:
            raise NotImplementedError
        self.model.set_optimiser(self.optimiser)
        self.batch_size = batch_size

    def train(self, X, y, epoch):
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        epoch_loss = 0.0
        batch_bar = tqdm(range(0, num_samples, self.batch_size),
                         desc=f"Epoch {epoch+1}", leave=False)
        for i in batch_bar:
            batch_indices = indices[i:i + self.batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            y_pred = self.model.forward(X_batch)
            train_loss = self.loss.forward(y_batch, y_pred)
            grad_loss = self.loss.backprop(y_batch, y_pred)
            self.model.zero_grad()
            self.model.backprop(grad_loss)
            self.model.step()
            epoch_loss += train_loss
            batch_bar.set_postfix(loss=train_loss)
        avg_epoch_loss = epoch_loss / (num_samples // self.batch_size)
        print(f"\nEpoch {epoch+1}, Avg Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def predict(self, X):
        y_pred = self.model.forward(X)
        y_label = np.argmax(y_pred, axis=1)
        return y_pred, y_label

    def eval(self, X, y):
        y_pred = self.model.forward(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y)
        val_loss = self.loss.forward(y, y_pred)
        return accuracy, val_loss
