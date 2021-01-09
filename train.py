import torch
from xray_dataset import load_dataset
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from resnet_model import get_model
from tqdm import tqdm

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)

    # Set hyperparameters
    batch_size =16
    num_epochs = 2

    # Load the data
    train_dataset, _ = load_dataset(stage='train', val_fraction=0.8)
    train_dataset.ready(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    # Load the model
    model = get_model().float().to(device)

    # Specify the optimizer and loss function
    loss_function = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}...')
        train_loss = 0
        for i, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch} train loss: {train_loss/i}')


