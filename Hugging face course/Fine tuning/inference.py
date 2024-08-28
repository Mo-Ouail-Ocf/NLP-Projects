from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pathlib import Path

# Define the path to the saved model and tokenizer
model_path = Path(__file__).parent / 'model/checkpoint-1377'
tokenizer_path = Path(__file__).parent / 'model'
""" model_path = Path('D:/nlp-projects/Hugging face course/Fine tuning/model/checkpoint-1377')
tokenizer_path = 'D:/nlp-projects/Hugging face course/Fine tuning/model' """

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def preprocess(sentence1, sentence2):
    # Tokenize the sentence pair
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding=True)
    return inputs

def predict(sentence1, sentence2):
    inputs = preprocess(sentence1, sentence2)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities

sentence1 = "I love learning new technologies"
sentence2 = "I admire AI learning "


probabilities = predict(sentence1, sentence2)
print(probabilities)

