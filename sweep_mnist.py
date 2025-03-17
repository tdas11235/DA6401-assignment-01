import data_loader as dl
import trainers as tr
import utils as ut
from tqdm import tqdm
import wandb
import yaml
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from config_mnist import configs


MNIST_CLASSES = [str(i) for i in range(10)]


def log_fmnist_images(x_train, y_train, project, entity_name='ch21b108-indian-institute-of-technology-madras'):
    wandb.init(entity=entity_name, project=project, name="sample_images")
    images = []
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        image_array = x_train[idx]
        images.append(wandb.Image(image_array, caption=MNIST_CLASSES[i]))
    wandb.log({"mnist data": images})
    wandb.finish()


def train(config, project, entity_name='ch21b108-indian-institute-of-technology-madras'):
    run_name = (
        f"hl_{config['hidden_layers']}"       # Number of hidden layers
        f"_hs_{config['hidden_size']}"        # Hidden layer sizes
        f"_bs_{config['batch_size']}"         # Batch size
        f"_ac_{config['activation']}"         # Activation function
        f"_opt_{config['optimizer']}"         # Optimizer
        f"_lr_{config['learning_rate']}"      # Learning rate
        f"_wd_{config['weight_decay']}"       # Weight decay
        f"_wi_{config['weight_init']}"        # Weight initialization
        f"_ep_{config['epochs']}"             # Number of epochs
    )
    wandb.init(entity=entity_name, project=project, name=run_name)
    hidden_sizes = [config['hidden_size']] * config['hidden_layers']
    trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=config['batch_size'],
        act_type=config['activation'],
        optimiser_type=config['optimizer'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        loss_type=loss_type
    )
    for epoch in tqdm(range(config['epochs']), desc="Training Progress"):
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
    wandb.finish()
    return val_accuracy


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(MNIST_CLASSES)))
    cm_normalized = np.nan_to_num(
        cm.astype('float') / cm.sum(axis=1, keepdims=True))
    customdata = np.dstack((cm, cm_normalized))
    fig = ff.create_annotated_heatmap(
        z=cm_normalized,
        x=MNIST_CLASSES,
        y=MNIST_CLASSES,
        annotation_text=[[f"{cm_normalized[i, j]:.1%}"
                          for j in range(len(MNIST_CLASSES))]
                         for i in range(len(MNIST_CLASSES))],
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
            tickvals=list(range(len(MNIST_CLASSES))),
            ticktext=MNIST_CLASSES
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(MNIST_CLASSES))),
            ticktext=MNIST_CLASSES
        ),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=800, height=800
    )
    wandb.log({"confusion_matrix_vis": wandb.Plotly(fig)})


def test_best(best_config, project, entity_name='ch21b108-indian-institute-of-technology-madras'):
    wandb.init(entity=entity_name, project=project,
               name="best_model")
    hidden_sizes = [best_config["hidden_size"]] * best_config["hidden_layers"]
    best_trainer = tr.StochasticTrainer(
        784, 10, hidden_sizes,
        batch_size=best_config["batch_size"],
        act_type=best_config["activation"],
        optimiser_type=best_config["optimizer"],
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"],
        loss_type=loss_type
    )
    for epoch in tqdm(range(best_config['epochs']), desc="Training Progress"):
        train_loss, _ = best_trainer.train(x_train, y_train, epoch)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
    test_accuracy, _ = best_trainer.eval(x_test, y_test)
    wandb.log({"test_accuracy": test_accuracy})
    _, y_pred_labels = best_trainer.predict(x_test)
    y_true_labels = np.argmax(y_test, axis=1)
    wandb.log({"confusion_matrix_data": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true_labels, preds=y_pred_labels, class_names=MNIST_CLASSES
    )})
    plot_confusion(y_true_labels, y_pred_labels)
    wandb.finish()


def main():
    global x_train, y_train, x_val, y_val, x_test, y_test, loss_type
    loss_type = "cross_entropy"
    project = "da6401-test-mnist-ce-3"
    (x_train, y_train), (x_test, y_test), (x_test_og,
                                           y_test_og) = dl.load_mnist_data()
    log_fmnist_images(x_test_og, y_test_og, project)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = ut.train_val_split(x_train, y_train)
    best_config = None
    mx = -1
    for config in configs:
        acc = train(config, project)
        if acc > mx:
            best_config = config
            mx = acc
    test_best(best_config, project)


if __name__ == '__main__':
    main()
