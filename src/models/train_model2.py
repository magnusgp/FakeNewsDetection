from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm
import torch
import numpy as np
from datasets import load_metric 
from copy import deepcopy
##from src.data.make_dataset import trainEx, testEx
from predict_model import *
import wandb
import pandas as pd
import numpy as np
import evaluate

wandb.init(project="mlops_fake_news", entity="ai_mark")
"""
wandb.config = {
  "lr": 5e-5,
  "nepochs": 10,
  "nsteps": 214
}
"""


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def train(accelerator = Accelerator(), lr=5e-5, nepoch=10, nsteps=214):
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}
    # load the pretrained model from a checkpoint
    checkpoint="roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Preparing model 
    dataset = torch.load('data/processed/dataset.pt')
    trainset = dataset['train']
    trainset = trainset.remove_columns(["text"]).rename_column('label', "labels").with_format("torch")
    testset = dataset['test']
    testset = testset.remove_columns(["text"]).rename_column('label', "labels").with_format("torch")
    # TODO: remove this, this is only to test the code
    trainset = trainset.select(range(0, 10))
    testset = testset.select(range(0, 10))
    
    training_args = TrainingArguments(
    output_dir="models/roberta-base",
    report_to="wandb",
    run_name="roberta-base",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
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
    train()