import pytest
import torch
from transformers import AutoModelForSequenceClassification


def test_model():
   checkpoint = "models/checkpoint-30"
   model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

   testset = torch.load('data/processed/testEx.pt')
   
   



   