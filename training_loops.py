import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def fit(model, train_loader, val_loader, optimizer, loss_function, train_metric_tracker, val_metric_tracker, early_stopping_tracker, max_epochs=1):
    """Training loop for a pytorch model."""
    for epoch in range(max_epochs):
        # Training
        for X, y in tqdm(train_loader):
            output = model(X)
            loss = loss_function(output, y)
            train_metric_tracker.add(output.item(), y.item(), loss.item())
            loss.backward()
            optimizer.step()
        # Validation
        with torch.no_grad():
            for X, y in tqdm(val_loader):
                output = model(X)
                loss = loss_function(output, y)
                val_metric_tracker.add(output.item(), y.item(), loss.item())
        # End of epoch
        train_metric_tracker.end_epoch()
        val_metric_tracker.end_epoch()
        # Check for early stopping condition
        if early_stopping_tracker(val_metric_tracker.loss_over_time[-1]):
            break


class EarlyStoppingTracker:
    """A class to keep track of the metric over time, stopping training when
    any chosen metric stops improving."""
    def __init__(self, patience=0, minimize=True):
        """Patience is the number of epochs to wait before pausing. The default
        of patience=0 means any time the metric fails to improve over the previous
        epoch, training will immediately halt.
        The minimize argument indicates whether the metric is being minimized
        or maximized."""
        self.patience = patience
        self.sign = (1 if minimize else -1)
        self.current_best = np.inf
        self.current_patience = self.patience

    def __call__(self, metric):
        if metric * self.sign < self.current_best:
            # Overwrite the previous best metric
            self.current_best = metric * self.sign
            # Reset patience counter
            self.current_patience = self.patience
        else:
            # No improvement to the metric, decrement patience.
            self.current_patience -= 1
            # If we've run out of patience
            if self.current_patience < 0:
                # Returns true to stop the training
                return True
        # Returns false to continue training
        return False


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class MetricTracker:
    def __init__(self):
        self.loss_over_time = []
        self.roc_auc_over_time = []
        self.per_class_roc_auc_over_time = []
        self.reset_storage()

    def reset_storage(self):
        self.loss = 0
        self.logits = []
        self.labels = []

    def roc_auc_score(self):
        all_labels = np.concatenate(self.labels)
        all_logits = np.concatenate(self.logits)
        all_probabilities = sigmoid(all_logits)
        return roc_auc_score(all_labels, all_probabilities, average=None)

    def end_epoch(self):
        self.loss_over_time.append(self.loss)
        per_class_roc_auc = self.roc_auc_score()
        self.per_class_roc_auc_over_time.append(per_class_roc_auc)
        self.roc_auc_over_time.append(np.mean(per_class_roc_auc))
        self.reset_storage()




