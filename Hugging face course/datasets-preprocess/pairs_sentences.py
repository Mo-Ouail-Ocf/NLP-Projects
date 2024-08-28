from transformers import AutoTokenizer , AutoModel
from transformers import DataCollatorWithPadding # final processing to form a batch
from datasets import load_dataset
from pprint import pprint
from torch.utils.data import DataLoader
# Natural language inference: wether 2 sentences are equiv : NLI

model_name='bert-base-cased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

""" pairs = [
    ['sentence 1','sentence 1'],
    ['sentence 2','sentence 2'],
]

pprint(tokenizer(pairs,padding='max_length',max_length=15)) 
"""

raw_dataset = load_dataset("glue", "mrpc")

train_raw_dataset = raw_dataset['train'] # sentence1,sentence2,label,idx

# Preprocessnig

def tokenizer_function(example):
    # left padding to false , pad dynamically to the batch size
    return tokenizer(example['sentence1'],example['sentence2'],truncation=True)

    #returns a dictionary with the keys 'input_ids', 'attention_mask','token_type_ids', 
    # those three fields are added to all splits of our dataset. 
train_ds = train_raw_dataset.map(tokenizer_function,batched=True)


### Dynamic PADDING with DataCollatorWithPadding 

data_collator = DataCollatorWithPadding(tokenizer)

samples = train_ds[:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
pprint([len(x) for x in samples["input_ids"]]) # Different legths 

batch = data_collator(samples) # applies padding with [PAD] with length  = max length in the returned batch
pprint({k: v.shape for k, v in batch.items()})


# Dynamic padding :
train_dl = DataLoader(
    dataset=train_ds,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator # pads all seqs in same batch to same length
)
# for gpus , cpus : better than fixed padding 

