from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from tqdm.auto import tqdm
import torch
import numpy as np
from datasets import load_metric 
from copy import deepcopy
#from src.data.make_dataset import trainloader, testloader
from predict_model import *

# Define model 
checkpoint="roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def train(accelerator, lr=5e-5, nepoch=10, nsteps=214):
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
    trainset = torch.load('data/processed/trainset.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model, optimizer, trainloader = accelerator.prepare(model, optim, trainloader)
    testloader = accelerator.prepare(testloader)
    
    best_val_acc = 0

    # Train model
    for epoch in range(nepoch):
        model.train()
        print(f"epoch n°{epoch+1}:")
        av_epoch_loss=0
        progress_bar = tqdm(range(nsteps))
        for batch in trainloader:
            #batch = {k:v.cuda() for k,v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            av_epoch_loss += loss
            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            predictions=torch.argmax(outputs.logits, dim=-1)
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
        val_acc = validate(model)
        if val_acc > best_val_acc:
            print("Achieved best validation accuracy so far. Saving model.")
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
        print("\n\n")

if __name__ == "__main__":
    accelerator = Accelerator()
    train(accelerator)
