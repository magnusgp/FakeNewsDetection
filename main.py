# Load the libraries
from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from joblib import load
import os

# Load the model
model_path = "models/roberta-base/checkpoint-30/"

# Load the roberta tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to the Fake News Detector API"}

# Define the route to the sentiment predictor
@app.post("/predict_if_fake_news/")
def predict(text_message):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
    classify = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer)
    return classify(text_message)