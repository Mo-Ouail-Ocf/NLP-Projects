from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model = model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentences = [
    'I am feeling good today',
    'I am very excited'
]

inputs = tokenizer(sentences, padding=True, return_tensors='pt')
inputs = {key: value.to('cuda') for key, value in inputs.items()}

predictions = model(**inputs)

print(predictions)
