from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_metric

##from src.data.make_dataset import trainEx, testEx
from predict_model import *
from tqdm.auto import tqdm
from transformers import AdamW, AutoModelForSequenceClassification

import wandb

# wandb.init(project="mlops_fake_news", entity="ai_mark")
"""
wandb.config = {
  "lr": 5e-5,
  "nepochs": 10,
  "nsteps": 214
}
"""


def train(accelerator=Accelerator(), lr=5e-5, nepoch=10, nsteps=214):
    # load the pretrained model from a checkpoint
    checkpoint = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # Define optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    # Accelerator pt.2
    device = accelerator.device
    model = model.to(device)
    print("We are using the following accelerator:" " ", device)

    # Load metric (f1 and accuracy)
    f1 = load_metric("f1")
    acc = load_metric("accuracy")

    # Preparing model
    dataset = torch.load("data/processed/dataset.pt")
    trainset = dataset["train"]
    trainset = (
        trainset.remove_columns(["text"])
        .rename_column("label", "labels")
        .with_format("torch")
    )
    trainset.select(range(0, 10))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model, optimizer, trainloader = accelerator.prepare(model, optim, trainloader)
    # testloader = accelerator.prepare(testloader)

    best_val_acc = 0

    # Train model
    for epoch in range(nepoch):
        model.train()
        print(f"epoch n°{epoch+1}:")
        av_epoch_loss = 0
        progress_bar = tqdm(range(nsteps))
        for batch in trainloader:
            print("Batch number: ", batch)
            # batch = {k:v.cuda() for k,v in batch.items()}
            optimizer.zero_grad()
            print("Im stuck when getting the outputs from the model")
            outputs = model(**batch)
            loss = outputs.loss
            # wandb.log({"loss": loss})
            av_epoch_loss += loss
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            predictions = torch.argmax(outputs.logits, dim=-1)
            f1.add_batch(predictions=predictions, references=batch["labels"])
            acc.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar.update(1)
        av_epoch_loss /= nsteps
        print(f"Training Loss: {av_epoch_loss: .2f}")
        acc_res = acc.compute()["accuracy"]
        print(f"Training Accuracy: {acc_res:.2f}")
        f_res = f1.compute()["f1"]
        print(f"Training F1-score: {f_res:.2f}")
        model.eval()
        """
        val_acc = validate(model)
        if val_acc > best_val_acc:
            print("Achieved best validation accuracy so far. Saving model.")
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
        print("\n\n")
        """


if __name__ == "__main__":
    train()