from datasets import load_dataset
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Load the MRPC dataset from GLUE
raw_datasets = load_dataset("glue", "mrpc")

# Get the train split of the dataset
train_raw_dataset = raw_datasets['train']

# Define the tokenization function
def tokenize_function(row):
    return tokenizer(
        row['sentence1'], row['sentence2'],
        padding='max_length', truncation=True, max_length=128,
    )

# Apply the tokenization function to the dataset
tokenized_ds = train_raw_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and rename the label column if needed
tokenized_ds = tokenized_ds.remove_columns(['sentence1', 'sentence2', 'idx'])

# Format the dataset for PyTorch
tokenized_ds = tokenized_ds.with_format('torch')

print(tokenized_ds[0])
