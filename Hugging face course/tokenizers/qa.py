from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch.nn.functional as F
import torch
from pprint import pprint

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries backed by 🤗 Transformers?"

inputs = tokenizer(question,context,return_tensors='pt') # 'input_ids' , 'attention_mask' ,''
sequence_ids = inputs.sequence_ids() # = token_types_ids  

outputs= model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

mask = [ token_id != 1 for token_id in sequence_ids]
# Unmask the [CLS] token , those where Mask = True will be masked
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask]=-10000
end_logits[mask]=-10000

start_probabilities = F.softmax(start_logits, dim=-1)[0] # (nb_token)
end_probabilities = F.softmax(end_logits, dim=-1)[0] # (nb_token)

# compute all possible products
scores = start_probabilities[:, None] * end_probabilities[None, :]
# mask entires where i>j
scores = torch.triu(scores)

# torch returns index in flattened tensor , use division & modulus
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index]) # of tokens

# Get the span in the text
inputs_with_offsets =  tokenizer(question,context,
                   return_offsets_mapping=True)
offset_mappings = inputs_with_offsets['offset_mapping']

sentence_start , _ = offset_mappings[start_index] 
_ , end = offset_mappings[end_index]

# you exist in the context :)
answer=context[sentence_start:end]
result = {
    "answer": answer,
    "start": sentence_start,
    "end": end,
    "score": scores[start_index, end_index],
}
pprint(result)

# Handling long context