import pennylane as qml
from pennylane import numpy as np


def hinge_loss(labels, predictions):
    """Compute hinge loss for labels in {-1, +1}."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    loss = np.maximum(0.0, 1.0 - labels * predictions)
    return np.mean(loss)


def accuracy(labels, predictions):
    """Compute binary accuracy from raw predictions using threshold at 0."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    preds = np.where(predictions >= 0.0, 1, -1)
    return np.mean(preds == labels)


def cost_function(weights, features, labels, quantum_model):
    """Evaluate dataset hinge loss for a given set of weights."""
    predictions = np.array([quantum_model(x, weights) for x in features])
    return hinge_loss(labels, predictions)


def predict_dataset(weights, features, quantum_model):
    """Run model inference over a whole dataset."""
    return np.array([quantum_model(x, weights) for x in features])


def to_pm_one_labels(labels):
    """Convert binary labels from {0, 1} to {-1, +1}."""
    labels = np.asarray(labels)
    return 2 * labels - 1


def current_lr(base_lr, global_step, optimizer_name, sgd_decay=0.0):
    """
    Learning-rate schedule used only for SGD to mimic decay from the paper.
    For Adam, returns the base learning rate unchanged.
    """
    if optimizer_name.lower() == "sgd" and sgd_decay > 0.0:
        return base_lr / (1.0 + sgd_decay * global_step)
    return base_lr


def make_optimizer(optimizer_name, lr, sgd_momentum=0.0):
    """
    Create one of the optimizers used in the paper:
    - adam
    - sgd
    If sgd_momentum > 0, MomentumOptimizer is used for the SGD case.
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return qml.AdamOptimizer(stepsize=lr)

    if optimizer_name == "sgd":
        if sgd_momentum > 0.0:
            return qml.MomentumOptimizer(stepsize=lr, momentum=sgd_momentum)
        return qml.GradientDescentOptimizer(stepsize=lr)

    raise ValueError(f"Unknown optimizer '{optimizer_name}'. Supported: adam, sgd")


def init_metrics_log():
    """Create a standard metrics dictionary used by training scripts."""
    return {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_mean_batch_loss": [],
        "effective_lr": [],
    }