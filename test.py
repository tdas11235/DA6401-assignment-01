import data_loader as dl
import trainers as tr
import utils as ut
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config):
    hidden_sizes = [config["hidden_size"]] * config["hidden_layers"]
    trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        act_type=config["activation"],
        optimiser_type=config["optimizer"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        loss_type='cross_entropy'
    )
    trainer.train(x_train, y_train)
    val_accuracy = trainer.eval(x_val, y_val)
    print(val_accuracy)


def main():
    global x_train, y_train, x_val, y_val
    (x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    config = {
        "batch_size": 128,
        "epochs": 10,
        "hidden_layers": 5,
        "hidden_size": 128,
        "learning_rate": 0.01,
        "optimizer": "nadam",
        "weight_decay": 0,
        "weight_init": "Xavier",
        "activation": "sigmoid"
    }
    train(config)


if __name__ == '__main__':
    main()
