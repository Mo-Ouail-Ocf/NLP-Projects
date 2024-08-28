from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_input = ['I am a little bit tired.', 'Hello everyone, have a nice day!']

inputs = tokenizer(raw_input,
                   padding=True,
                   truncation=True,
                   return_tensors="pt")

# inputs : 'input_ids':tensor[batch_size,seq+padding size] , 'mask':[batch_size,seq+padding size]

outputs = model(**inputs)

print(model.config.id2label)

print(outputs.logits.shape)
