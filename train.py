import torch
from xray_dataset import load_dataset
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from resnet_model import get_model
from tqdm import tqdm

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set hyperparameters
    batch_size = 64
    num_epochs = 2

    # Load the data
    train_dataset, val_dataset = load_dataset(stage='train', val_fraction=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Load the model
    model = get_model().float().to(device)

    # Specify the optimizer and loss function
    loss_function = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}...')
        train_loss = 0
        for i, (X, y) in tqdm(enumerate(train_loader),total=len(train_loader)):
            if (i==0):
                print(X.dtype)
                print(y.dtype)
            X = X.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch} train loss: {train_loss/i}')


