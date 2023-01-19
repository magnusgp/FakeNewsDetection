import os

import pytest
import torch
from torch import nn, optim
from tqdm import tqdm

from src.models.model import Model


@pytest.mark.skipif(not os.path.exists('data/processed/trainset.pt'), reason="Training files not found")
@pytest.mark.parametrize("lr,criterion", [(1e-3, nn.NLLLoss()), (1e-4, nn.CrossEntropyLoss()), (1e-5, nn.NLLLoss())])
def test_training(lr, criterion):
    epochs = 5
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = criterion
    # load trainset from data folder
    trainset = torch.load('data/processed/trainset.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    training_loss = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # check that the training loss is not negative
            assert running_loss > 0, "Training loss is negative"
            break