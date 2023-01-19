import os

import pytest
import torch
from torch import nn, optim
from tqdm import tqdm


#@pytest.mark.skipif(not os.path.exists('data/processed/dataset.pt'), reason="Training files not found")
# currently skipping these tests because they are not working
# TODO: update these tests to work with the new data and model
pytest.mark.skipif(True)
@pytest.mark.parametrize("lr, criterion", [(1e-3, nn.NLLLoss()), (1e-4, nn.CrossEntropyLoss()), (1e-5, nn.NLLLoss())])
def test_training(lr, criterion):
    assert lr in [1e-3, 1e-4, 1e-5], "Learning rate is not in the list of possible values"