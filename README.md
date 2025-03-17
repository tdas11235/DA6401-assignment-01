# DA6401 : Assignment-1

## Code Structure

### Neural Network Files

1. `neural_net.py`: This module defines the different activation functions, a `Linear` layer with forward pass and backpropagation and a `Module` class that collected and modifies all the paramters in the neural network.
   - `Linear` class: This has different modes of initialization of weights and biases. The forward pass is implemented in the `forward()` method and backpropagation in the `backprop()` method.
   - `Module` class: This class is meant to collect the values of all the parameters of the model. The main methods are `parameters()` that return the parameters of the model (although this was not used in the optimizers), `step()` to update all the parameters at once via the optimizer, `zero_grad()` to set gradients of paramaters to 0 and `set_optimiser()` to set an optimiser for the neural network.
2. `deepnn.py`: This mainly is to have the `DNN` class which creates a deep neural network with hidden layers. The activation type is also taken as an input for this class.
3. `losses.py`: This file has the backpropagation and forward pass for cross-entropy loss and mean-squared loss with the activation of the output layer being assumed to be softmax.
4. `optimisers.py`: This file has the code for all the optimisers. All the optimisers are written over one base class. This makes it easy to integrate any new optimiser with the current codebase.

### Dataset and split files

- `data_loader.py`: This module contains the code to load the train and test splits of both the MNIST dataset and the Fashion MNIST dataset.
- `utils.py`: This module contains the code to do a train-val split of the data based on a split fraction and a numpy seed (for uniformity).

### Training files

1. `trainers.py`: This file defines two trainers.
   - `NormalTrainer` class: This trains the neural network on the whole training data instead of batches per epoch.
   - `StochasticTrainer` class: This implements the stochastic version of the training algorithm. The `__init__()` method sets up the loss function and the optimiser for the network. `train()` method sets up the training loop by dividing the data into batches of mentioned size. Then we have `predict()` to do a single forward pass over the network for predictions given input and `eval()` to compute loss and accuracy for inference.
2. `train.py`: This file is made according to Q10 of the assignment. This takes in the commandline arguments and trains the network and also predicts on the test split while logging everything to the wandb dashboard.

### Sweep files:

1. `sweep.py`: To conduct a sweep over Fashion MNIST data using cross-entropy loss.
2. `sweep_mse.py`: To conduct a sweep over Fashion MNIST data using mean-squared loss.
3. `sweep_mnist.py`: To conduct a sweep over MNIST data using cross-entropy loss.
4. `sweep_mse_mnist.py`: To conduct a sweep over MNIST data using mean-squared loss.
5. `config.yaml`: The configuration for wandb sweeps for files (1) and (2)
6. `config_mnist.py`: The configuration for wandb sweeps for files (3) and (4)

## Some important points

1. I have found that the training is more stable when the biases are not randomly initialized but kept as 0s. So, I have trained my network with this assumption. The weights are still initialized according to the schemes given. There is an option to initialize biases randomly too if we change `init_b=True` in the `Linear` class.
2. The SGD is implemented as mini-batch stochastic gradient descent for hyperparameter sweeps as SGD takes one data point to do backpropagation and loops over all the datapoints in an epoch which takes a lot of time and appears infeasible. But, we can still run SGD with batch_size=1 by setting `true_sgd=True` in `StochasticTrainer`.

## Commands

Install dependencies

```bash
pip install -r requirements.txt
```

Login to wand

```bash
wandb login
```

To run the `train.py` file

```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

## Links

- Link to Report: [Report](https://api.wandb.ai/links/ch21b108-indian-institute-of-technology-madras/nilg5zh5)

- Link to GitHub: [Codebase](https://github.com/tdas11235/DA6401-assignment-01)
