import os.path

import pytest
import torch


@pytest.mark.skipif(
    not os.path.exists("data/processed/dataset.pt"), reason="Training files not found"
)
def test_data():
    # load data from data/processed folder with pytorch
    dataset = torch.load("data/processed/dataset.pt")

    # create the train and test sets from the dataset
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
    assert len(trainset) > 0, "Training set is empty"
    assert len(testset) > 0, "Test set is empty"

    # assert that there are both true and fake labels in the training and test sets
    # get all labels from the training and test sets
    train_labels = [trainset[i]["labels"] for i in range(len(trainset))]
    test_labels = [testset[i]["labels"] for i in range(len(testset))]

    labels = [0, 1]

    for label in labels:
        assert label in train_labels, "Training set does not contain label {}".format(
            label
        )
        assert label in test_labels, "Test set does not contain label {}".format(label)


if __name__ == "__main__":
    test_data()
