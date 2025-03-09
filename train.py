import argparse
import data_loader as dl
import trainers as tr
import utils as ut
import wandb
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="train with one set of hyperparams")
    parser.add_argument("-wp", "--wandb_project", type=str, default="ch21b108-indian-institute-of-technology-madras",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="train",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use. Choices: ['mnist', 'fashion_mnist']")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size used to train the neural network.")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",
                        help="Loss function. Choices: ['mean_squared_error', 'cross_entropy']")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam",
                        help="Optimizer. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.9,
                        help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float,
                        default=0.9, help="Beta used by RMSProp optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                        help="Beta1 used by Adam and Nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                        help="Beta2 used by Adam and Nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float,
                        default=1e-8, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float,
                        default=0.0, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier",
                        help="Weight initialization method. Choices: ['random', 'Xavier']")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5,
                        help="Number of hidden layers in the feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid",
                        help="Activation function. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU']")
    return parser.parse_args()


def train(config):
    wandb.init(entity=config.wandb_entity, project=config.wandb_project)
    hidden_sizes = [config.hidden_size] * config.num_layers
    run_name = (
        f"hl_{config.num_layers}"
        f"_hs_{config.hidden_size}"
        f"_bs_{config.batch_size}"
        f"_ac_{config.activation}"
        f"_opt_{config.optimizer}"
        f"_lr_{config.learning_rate}"
        f"_wd_{config.weight_decay}"
        f"_wi_{config.weight_init}"
        f"_ep_{config.epochs}"
    )
    wandb.run.name = run_name
    wandb.run.save()
    trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        optimiser_type=config.optimizer,
        momentum=config.momentum,
        beta=config.beta,
        beta1=config.beta1,
        beta2=config.beta2,
        act_type=config.activation,
        loss_type=config.loss,
        init_method=config.weight_init,
        weight_decay=config.weight_decay

    )
    for epoch in tqdm(range(config.epochs), desc="Training Progress"):
        train_loss = trainer.train(x_train, y_train, epoch)
        val_accuracy, val_loss = trainer.eval(x_val, y_val)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    final_accuracy, _ = trainer.eval(x_val, y_val)
    wandb.log({"accuracy": final_accuracy})


def main(args):
    global x_train, y_train, x_val, y_val
    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = dl.load_mnist_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    train(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
