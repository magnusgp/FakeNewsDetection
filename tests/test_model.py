import pytest
import torch
from transformers import AutoModelForSequenceClassification

def test_model():
    # test test
    checkpoint = "models/checkpoint-30"

    dataset = torch.load('data/processed/dataset.pt')

    # test that the model outputs the correct shape
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.eval()
    input_ids = dataset['test']['input_ids']
    attention_mask = dataset['test']['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)
    assert output.logits.shape == (100, 2)
