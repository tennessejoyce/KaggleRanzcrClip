import torch
from xray_dataset import load_dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from architectures import EfficientNetB0
from torch.utils.data import DataLoader
from training_loops import *
from time import time

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f'Running on {device}')

    # Set hyperparameters
    batch_size = 50
    patience = 3
    max_epochs = 100
    data_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}

    # Load the data
    print('Loading dataset...')
    train_dataset, val_dataset = load_dataset(stage='train', drop_fraction=0, val_type='split', val_fraction=0.2)
    train_loader = DataLoader(train_dataset, **data_loader_kwargs)
    val_loader = DataLoader(val_dataset, **data_loader_kwargs)


    # Load the model
    print('Loading model...')
    model = EfficientNetB0().float().to(device)

    # Specify the optimizer and loss function
    print('Setting hyperparameters...')
    loss_function = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    early_stopping_tracker = EarlyStoppingTracker(patience=patience, minimize=False, saved_model_file=f'saved_models/{int(time())}.pt')
    train_metric_tracker = MetricTracker(name='train')
    val_metric_tracker = MetricTracker(name='validation')

    print('Training model...')
    fit(model, train_loader, val_loader, optimizer, loss_function, train_metric_tracker,
        val_metric_tracker, early_stopping_tracker, max_epochs=max_epochs)

    print('Finished training')
