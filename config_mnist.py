configs = [
    {
        "hidden_size": 64,
        "hidden_layers": 4,
        "batch_size": 16,
        "activation": "ReLU",
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "weight_init": "Xavier",
        "epochs": 10
    },
    {
        "hidden_size": 64,
        "hidden_layers": 3,
        "batch_size": 16,
        "activation": "ReLU",
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "weight_init": "Xavier",
        "epochs": 10
    },
    {
        "hidden_size": 64,
        "hidden_layers": 4,
        "batch_size": 32,
        "activation": "tanh",
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "weight_init": "Xavier",
        "epochs": 10
    },
]

configs_mse = [
    {
        "hidden_size": 64,
        "hidden_layers": 4,
        "batch_size": 16,
        "activation": "tanh",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "weight_init": "Xavier",
        "epochs": 10
    },
    {
        "hidden_size": 128,
        "hidden_layers": 4,
        "batch_size": 16,
        "activation": "tanh",
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "weight_init": "Xavier",
        "epochs": 10
    },
    {
        "hidden_size": 64,
        "hidden_layers": 4,
        "batch_size": 16,
        "activation": "ReLU",
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "weight_init": "Xavier",
        "epochs": 10
    },
]
