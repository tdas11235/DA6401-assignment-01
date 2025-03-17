import numpy as np


def train_val_split(X, y, ratio=0.1, seed=42):
    np.random.seed(seed)
    class_labels = np.argmax(y, axis=1)
    unique_classes = np.unique(class_labels)
    X_train_list, X_val_list = [], []
    y_train_list, y_val_list = [], []
    # shuffle and pick from classes individually
    for cls in unique_classes:
        class_indices = np.where(class_labels == cls)[0]
        np.random.shuffle(class_indices)
        split = int(len(class_indices) * (1 - ratio))
        X_train_list.append(X[class_indices[:split]])
        X_val_list.append(X[class_indices[split:]])
        y_train_list.append(y[class_indices[:split]])
        y_val_list.append(y[class_indices[split:]])
    X_train, X_val = np.concatenate(X_train_list), np.concatenate(X_val_list)
    y_train, y_val = np.concatenate(y_train_list), np.concatenate(y_val_list)
    train_indices = np.arange(X_train.shape[0])
    val_indices = np.arange(X_val.shape[0])
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return X_train[train_indices], X_val[val_indices], y_train[train_indices], y_val[val_indices]
