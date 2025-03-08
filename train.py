import data_loader as dl
import trainers as tr
import utils as ut
import wandb
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train():
    wandb.init()
    config = wandb.config
    hidden_sizes = [config.hidden_size] * config.hidden_layers
    run_name = (
        f"hl_{config.hidden_layers}"       # Number of hidden layers
        f"_hs_{config.hidden_size}"        # Hidden layer sizes
        f"_bs_{config.batch_size}"         # Batch size
        f"_ac_{config.activation}"         # Activation function
        f"_opt_{config.optimizer}"         # Optimizer
        f"_lr_{config.learning_rate}"      # Learning rate
        f"_wd_{config.weight_decay}"       # Weight decay
        f"_wi_{config.weight_init}"        # Weight initialization
        f"_ep_{config.epochs}"             # Number of epochs
    )
    wandb.run.name = run_name
    wandb.run.save()
    trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=config.batch_size,
        epochs=config.epochs,
        act_type=config.activation,
        optimiser_type=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        loss_type='cross_entropy'
    )
    trainer.train(x_train, y_train)
    val_accuracy = trainer.eval(x_val, y_val)
    wandb.log({"val_accuracy": val_accuracy})


def main():
    sweep_config = load_config()
    global x_train, y_train, x_val, y_val
    (x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    sweep_id = wandb.sweep(sweep_config, project="da6401-test-2")
    wandb.agent(sweep_id, train, count=40)


if __name__ == '__main__':
    main()
