import torch
from accelerate import Accelerator
from datasets import load_metric
from transformers import AutoModelForSequenceClassification

# Load metric (f1 and accuracy)
f1 = load_metric("f1")
acc = load_metric("accuracy")

# Accelerator 
accelerator = Accelerator()

#Validate model 
def validate():
    checkpoint="roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    testset = torch.load('data/processed/testEx.pt')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    for batch in testloader:
        #batch = {k:v.cuda() for k,v in batch.items()}
        outputs = model(**batch)
        predictions=torch.argmax(outputs.logits, dim=-1)
        #f1.add_batch(predictions=predictions, references=batch["labels"])
        f1.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))
        #acc.add_batch(predictions=predictions, references=batch["labels"])
        acc.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))
    acc_res = acc.compute()["accuracy"]
    print(f"Validation Accuracy: {acc_res:.2f}")
    f_res = f1.compute()["f1"]
    print(f"Validation F1-score: {f_res:.2f}")
    return acc_res

if __name__ == "__main__":
    validate()
    