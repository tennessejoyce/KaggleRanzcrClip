import torch
from tqdm import tqdm
import numpy as np


def fit(model, train_loader, val_loader, optimizer, loss_function, metric_tracker, early_stopping_tracker, max_epochs=1):
    """General-purpose training loop for a pytorch model."""
    for epoch in range(max_epochs):
        # Training
        for X, y in tqdm(train_loader):
            output = model(X)
            loss = loss_function(output, y)
            metric_tracker.add_train(output.item(), y.item(), loss.item())
            loss.backward()
            optimizer.step()
        # Validation
        with torch.no_grad():
            for X, y in tqdm(val_loader):
                output = model(X)
                loss = loss_function(output, y)
                metric_tracker.add_val(output.item(), y.item(), loss.item())
        metric_tracker.end_epoch(epoch)
        if early_stopping_tracker(metric_tracker.val_metric[-1]):
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


class MetricTracker:
    def __init__(self):
        pass




