from train_model import classifier

# Data input
sequence_to_classify = "This is fake news"
candidate_labels = ['True', 'Fake']
predictions = []
if classifier(sequence_to_classify, candidate_labels).get('scores')[0] > 0.5:
    predictions.append(1)
else:
    predictions.append(0)

print(predictions)

