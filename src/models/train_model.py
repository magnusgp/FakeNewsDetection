from copy import deepcopy

import evaluate
import hydra
import numpy as np
import torch
from datasets import load_metric

from predict_model import *
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


@hydra.main(config_path="config", config_name="config.yaml")
def train(config):
    '''Function that takes the hydra config as input and trains the model using these parameters
    The function is based on the example from the transformers library:
    www.github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification.
    The model outputs checkpoints and logs to wandb.'''
    
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}
    # load the pretrained model from a checkpoint
    params = config.experiments
    checkpoint = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preparing model
    dataset = torch.load(
        "/Users/magnus/Desktop/DTU/5semester/MLOPS/TrueOrFakeNews/data/processed/dataset.pt"
    )
    # Make sure that we only use the encoded data, both for training and testing
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
    # Subsample of the data to train the model locally
    # Can be removed when training on the full dataset in cloud
    trainset = trainset.select(range(0, 100))
    testset = testset.select(range(0, 100))

    # test/assert that the params are of correct type
    stringparams = [
        "output_dir",
        "report_to",
        "run_name",
        "evaluation_strategy",
        "save_strategy",
    ]
    floatparams = ["learning_rate", "weight_decay"]
    intparams = [
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "num_train_epochs",
        "load_best_model_at_end",
    ]
    for param in params:
        if param in stringparams:
            assert isinstance(params[param], str), f"{param} must be a string"
        elif param in floatparams:
            assert isinstance(params[param], float), f"{param} must be a float"
        elif param in intparams:
            assert isinstance(params[param], int), f"{param} must be an int"
        else:
            raise ValueError(f"Parameter type of parameter: {param} is not recognized!")

    # Pass arguments to the trainer
    training_args = TrainingArguments(
        output_dir=params["output_dir"],
        report_to=params["report_to"],
        run_name=params["run_name"],
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        per_device_eval_batch_size=params["per_device_eval_batch_size"],
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        evaluation_strategy=params["evaluation_strategy"],
        save_strategy=params["save_strategy"],
        load_best_model_at_end=params["load_best_model_at_end"],
        daatloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("So far so good!")
    # Train the model
    trainer.train()


if __name__ == "__main__":
    train()
