import data_loader as dl
import trainers as tr
import utils as ut
from tqdm import tqdm
import wandb
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import argparse

F_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MNIST_CLASSES = [str(i) for i in range(10)]


def get_args():
    parser = argparse.ArgumentParser(
        description="train with one set of hyperparams")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401-train",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ch21b108-indian-institute-of-technology-madras",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use. Choices: ['mnist', 'fashion_mnist']")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
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
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random",
                        help="Weight initialization method. Choices: ['random', 'Xavier']")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers in the feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=64,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU",
                        help="Activation function. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU']")
    return parser.parse_args()


def log_fmnist_images(x_train, y_train, project, entity_name='ch21b108-indian-institute-of-technology-madras'):
    wandb.init(entity=entity_name, project=project, name="sample_images")
    images = []
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        image_array = x_train[idx]
        images.append(wandb.Image(image_array, caption=F_MNIST_CLASSES[i]))
    wandb.log({"Fashion_mnist data": images})
    wandb.finish()


def log_mnist_images(x_train, y_train, project, entity_name='ch21b108-indian-institute-of-technology-madras'):
    wandb.init(entity=entity_name, project=project, name="sample_images")
    images = []
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        image_array = x_train[idx]
        images.append(wandb.Image(image_array, caption=MNIST_CLASSES[i]))
    wandb.log({"mnist data": images})
    wandb.finish()


def train(config, project, entity_name, CLASSES):
    run_name = (
        f"hl_{config.num_layers}"       # Number of hidden layers
        f"_hs_{config.hidden_size}"        # Hidden layer sizes
        f"_bs_{config.batch_size}"         # Batch size
        f"_ac_{config.activation}"         # Activation function
        f"_opt_{config.optimizer}"         # Optimizer
        f"_lr_{config.learning_rate}"      # Learning rate
        f"_wd_{config.weight_decay}"       # Weight decay
        f"_wi_{config.weight_init}"        # Weight initialization
        f"_ep_{config.epochs}"             # Number of epochs
    )
    wandb.init(entity=entity_name, project=project, name=run_name)
    hidden_sizes = [config.hidden_size] * config.num_layers
    trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=config.batch_size,
        act_type=config.activation,
        optimiser_type=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        loss_type=loss_type
    )
    for epoch in tqdm(range(config.epochs), desc="Training Progress"):
        train_loss, train_acc = trainer.train(x_train, y_train, epoch)
        val_accuracy, val_loss = trainer.eval(x_val, y_val)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    val_accuracy, _ = trainer.eval(x_val, y_val)
    wandb.log({"accuracy": val_accuracy})
    test(trainer, CLASSES)
    wandb.finish()


def plot_confusion(y_true, y_pred, CLASSES):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cm_normalized = np.nan_to_num(
        cm.astype('float') / cm.sum(axis=1, keepdims=True))
    customdata = np.dstack((cm, cm_normalized))
    fig = ff.create_annotated_heatmap(
        z=cm_normalized,
        x=CLASSES,
        y=CLASSES,
        annotation_text=[[f"{cm_normalized[i, j]:.1%}"
                          for j in range(len(CLASSES))]
                         for i in range(len(CLASSES))],
        colorscale=[
            [0.0, "red"],
            [0.4, "yellow"],
            [1.0, "green"]
        ],
        showscale=True,
        hoverinfo="text"
    )
    fig.update_traces(
        customdata=customdata,
        name='',
        hovertemplate=(
            "<b>True: %{y}</b><br>"
            "<b>Pred: %{x}</b><br>"
            "Count: %{customdata[0]:.0f}<br>"
            "Ratio: %{customdata[1]:.1%}"
        )
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(CLASSES))),
            ticktext=CLASSES
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(CLASSES))),
            ticktext=CLASSES
        ),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=800, height=800
    )
    wandb.log({"confusion_matrix_vis": wandb.Plotly(fig)})


def test(trainer, CLASSES):
    test_accuracy, _ = trainer.eval(x_test, y_test)
    wandb.log({"test_accuracy": test_accuracy})
    _, y_pred_labels = trainer.predict(x_test)
    y_true_labels = np.argmax(y_test, axis=1)
    wandb.log({"confusion_matrix_data": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true_labels, preds=y_pred_labels, class_names=CLASSES
    )})
    plot_confusion(y_true_labels, y_pred_labels, CLASSES)


def main(args):
    global x_train, y_train, x_val, y_val, x_test, y_test, loss_type
    loss_type = args.loss
    project = args.wandb_project
    entity_name = args.wandb_entity
    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test), (x_test_og,
                                               y_test_og) = dl.load_fashion_data()
        CLASSES = F_MNIST_CLASSES
        log_fmnist_images(x_test_og, y_test_og, project, entity_name)
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test), (x_test_og,
                                               y_test_og) = dl.load_mnist_data()
        CLASSES = MNIST_CLASSES
        log_mnist_images(x_test_og, y_test_og, project, entity_name)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    train(args, project, entity_name, CLASSES)


if __name__ == "__main__":
    args = get_args()
    main(args)
