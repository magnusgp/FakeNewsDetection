import pytest
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@pytest.mark.skipif(not os.path.exists('data/processed/dataset.pt'), reason="Training files not found")
def test_model():
    # test test
    checkpoint = "models/roberta-base/checkpoint-30"
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    dataset = torch.load('data/processed/dataset.pt')

    # test that the model outputs the correct shape
    classify = pipeline("text-classification", model=checkpoint, tokenizer=tokenizer)
    output = classify("This is a test")
    
    # Assert that the model outputs a dictionary
    assert isinstance(output[0], dict)
    
    # Assert that the label that is outputted is either FAKE or REAL
    assert output[0]["label"] in ["FAKE", "REAL"]
