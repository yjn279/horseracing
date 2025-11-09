import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import get_datasets
from model import HorseRacingModel
from train import train
from test import test
from utils import get_device

def main():
    # Get device
    device = get_device()
    
    # Get datasets
    train_dataset, test_dataset = get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    model = HorseRacingModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train and test
    epochs = 50
    for i in range(epochs):
        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, device)
            test(test_dataloader, model, loss_fn, device)


if __name__ == '__main__':
    main()
