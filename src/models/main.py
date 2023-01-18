from src.models.train_model import train
from src.models.predict_model import validate
from transformers import AutoModelForSequenceClassification, AdamW
from accelerate import Accelerator

if __name__ == "__main__":
    # Accelerator 
    accelerator = Accelerator()
    
    # Define model 
    checkpoint="roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    train(accelerator, model=model, lr=5e-5, nepoch=10, nsteps=214)
    
    validate(model)