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
    def __init__(self, in_dim, out_dim, hidden_dims, lr=0.001, optimiser_type='NAG', beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='ce', epochs=100, init_method="random"):
        self.model = DNN(in_dim, out_dim, hidden_dims, act_type, init_method)
        if loss_type == 'ce':
            self.loss = CrossEntropy()
        else:
            self.loss = MSE()
        if optimiser_type == "GD":
            self.optimiser = opt.GD(lr)
        elif optimiser_type == "MGD":
            self.optimiser = opt.MGD(lr, beta)
        elif optimiser_type == "NAG":
            self.optimiser = opt.NAG(lr, beta)
        elif optimiser_type == "RMSProp":
            self.optimiser = opt.RMSProp(lr, beta)
        elif optimiser_type == "Adam":
            self.optimiser = opt.Adam(lr, beta1, beta2)
        elif optimiser_type == "Nadam":
            self.optimiser = opt.Nadam(lr, beta1, beta2)
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
    def __init__(self, in_dim, out_dim, hidden_dims, batch_size=1, lr=0.001, optimiser_type='NAG', beta=0.9, beta1=0.9, beta2=0.999, act_type='sigmoid', loss_type='ce', epochs=100, init_method="random"):
        self.model = DNN(in_dim, out_dim, hidden_dims, act_type, init_method)
        if loss_type == 'ce':
            self.loss = CrossEntropy()
        else:
            self.loss = MSE()
        if optimiser_type == "GD":
            self.optimiser = opt.GD(lr)
        elif optimiser_type == "MGD":
            self.optimiser = opt.MGD(lr, beta)
        elif optimiser_type == "NAG":
            self.optimiser = opt.NAG(lr, beta)
        elif optimiser_type == "RMSProp":
            self.optimiser = opt.RMSProp(lr, beta)
        elif optimiser_type == "Adam":
            self.optimiser = opt.Adam(lr, beta1, beta2)
        elif optimiser_type == "Nadam":
            self.optimiser = opt.Nadam(lr, beta1, beta2)
        else:
            raise NotImplementedError
        self.model.set_optimiser(self.optimiser)
        self.epochs = epochs
        self.bacth_size = batch_size

    def train(self, X, y):
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
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

    def eval(self, X, y):
        y_pred = self.model.forward(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy
