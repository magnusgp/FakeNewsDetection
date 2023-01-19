import os.path

import numpy as np
import pytest
import torch


@pytest.mark.skipif(not os.path.exists('data/processed/dataset.pt'), reason="Training files not found")
def test_data():
    # load data from data/processed folder with pytorch
    dataset = torch.load(
        "processed/dataset.pt"
    )

    trainset = dataset["train"]
    trainset = (
        trainset.remove_columns(["text"])
        .rename_column("label", "labels")
        .with_format("torch")
    )
    testset = dataset["test"]
    testset = (
        testset.remove_columns(["text"])
        .rename_column("label", "labels")
        .with_format("torch")
    )
    
    # check that the training and test sets are not empty
    assert trainset.size() != 0, "Training set is empty"
    assert testset.size() != 0, "Test set is empty"
    
    # check that the input data is a string of text inside of a tensor
    assert isinstance(trainset[0][0], torch.Tensor), "Input data is not a tensor"
    
    # assert that there are both true and fake labels in the training and test sets
    # get all labels from the training and test sets
    train_labels = [trainset[i][1].item() for i in range(len(trainset))]
    test_labels = [testset[i][1].item() for i in range(len(testset))]
    
    labels = [0, 1]
    
    for label in labels:
        assert label in train_labels, "Training set does not contain label {}".format(label)
        assert label in test_labels, "Test set does not contain label {}".format(label)
    
    
    