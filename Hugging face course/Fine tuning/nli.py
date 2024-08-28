from transformers import AutoModelForSequenceClassification,AutoTokenizer,\
    DataCollatorWithPadding,Trainer,TrainingArguments

import evaluate

from datasets import load_dataset

from pathlib import Path

import numpy as np

CHECKPOINT = 'bert-base-cased'

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,num_labels=2)

raw_dataset = load_dataset('glue','mrpc')

def tokenizer_function(row):
    # set padding to default False to use dynamic padding using the collator
    return tokenizer(
        row['sentence1'],
        row['sentence2'],
        truncation=True
    )

tokenized_ds = raw_dataset.map(tokenizer_function)
data_collator = DataCollatorWithPadding(tokenizer)

# Hyperparams

path = Path(__file__).parent / 'model'
training_args=TrainingArguments(
    output_dir=str(path),push_to_hub=True,
    eval_strategy='epoch',
    
    )

# For evaluation provide :
# 1- Eval strategy : after each epoch
# 2- metrics to be computed

# Evaluation :

# 1- compute_metrics() : EvalPrediction : [predictions,label_ids ] - >{'metric':value}

def compute_metrics(eval_preds):
    # load metrics attached to the dataset
    metric = evaluate.load("glue","mrpc")
    logits , labels = eval_preds
    preds = np.argmax(logits,axis=-1)
    return metric.compute(predictions=preds,references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    compute_metrics=compute_metrics,
)

if __name__=="__main__":
    trainer.train()

    