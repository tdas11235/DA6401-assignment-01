import data_loader as dl
import trainers as tr
import utils as ut
from tqdm import tqdm
import wandb
import yaml
import numpy as np


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def log_images(x_train, y_train):
    wandb.init(project="da6401-test-3", name="sample_images")
    class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    images = []
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        image_array = x_train[idx]
        images.append(wandb.Image(image_array, caption=class_labels[i]))
    wandb.log({"dataset": images})
    wandb.finish()


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
        act_type=config.activation,
        optimiser_type=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        loss_type='cross_entropy'
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
    val_accuracy, _ = trainer.eval(x_val, y_val)
    wandb.log({"accuracy": val_accuracy})


def main():
    sweep_config = load_config()
    global x_train, y_train, x_val, y_val
    (x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
    log_images(x_train, y_train)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    sweep_id = wandb.sweep(sweep_config, project="da6401-test-3")
    wandb.agent(sweep_id, train, count=30)


if __name__ == '__main__':
    main()
