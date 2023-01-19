from copy import deepcopy

import evaluate
import hydra
import os
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_metric

##from src.data.make_dataset import trainEx, testEx
from predict_model import *
from tqdm.auto import tqdm
from transformers import (
    AdamW,
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
    # dataset = torch.load('data/processed/dataset.pt')
    print(os.getcwd())
    print(os.path.abspath(os.getcwd()))
    print(r"{}/data/processed/dataset.csv".format((os.path.abspath(os.path.abspath(os.path.abspath(os.path.abspath(os.getcwd())))))))
    dataset = torch.load(r"{}/data/processed/dataset.csv".format((os.path.abspath(os.path.abspath(os.path.abspath(os.path.abspath(os.getcwd())))))))
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
    # TODO: remove this, this is only to test the code
    trainset = trainset.select(range(0, 100))
    testset = testset.select(range(0, 100))

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

    trainer.train()


if __name__ == "__main__":
    # wandb.init(project="mlops_fake_news", entity="ai_mark")
    train()
