from transformers import pipeline

# Load the model
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')


