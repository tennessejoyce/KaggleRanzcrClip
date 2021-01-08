import torch
from xray_dataset import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':

    batch_size = 64

    train_dataset, val_dataset = load_dataset(stage='train', validation_fraction=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

