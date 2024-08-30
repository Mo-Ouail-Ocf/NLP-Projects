from datasets import load_dataset   
from pathlib import Path
from pprint import pprint
import html
from transformers import AutoTokenizer

if __name__ == "__main__":
    data_dir = Path(__file__).parent / 'drugsCom_raw'

    train_path = data_dir / 'drugsComTrain_raw.tsv'
    test_path = data_dir / 'drugsComTest_raw.tsv'

    data_files = {
        'train': str(train_path),
        'test': str(test_path)
    }

    raw_ds = load_dataset('csv', data_files=data_files, delimiter='\t')

    # Renaming, cleaning & filtering

    # Rename
    raw_ds = raw_ds.rename_column(
        original_column_name="Unnamed: 0",
        new_column_name="patient_id"
    )

    # Clean null condition data
    raw_ds = raw_ds.filter(
        lambda row: row["condition"] is not None,
        num_proc=4
    )

    # Normalize condition column
    def lower_case_condition(batch):
        return {
            "condition": [condition.lower() for condition in batch["condition"]]
        }
    raw_ds = raw_ds.map(lower_case_condition, batched=True, num_proc=4)

    # Adding length of review cols
    def add_review_len(batch):
        return {
            'review_length': [len(review.split()) for review in batch['review']]
        }

    raw_ds = raw_ds.map(add_review_len, batched=True, num_proc=4)

    # Remove reviews that contain fewer than 30 words
    def filter_review_length(batch):
        return [length > 30 for length in batch['review_length']]

    raw_ds = raw_ds.filter(filter_review_length, batched=True, num_proc=4)

    # Unescape HTML entities in reviews
    def unescape_html_entities(batch):
        return {'review': [html.unescape(review) for review in batch['review']]}
    
    raw_ds = raw_ds.map(unescape_html_entities, batched=True, num_proc=4)

    # Tokenize 

    # this function will adds more examples than the original dataset
    # Solution :
    
    # 1- tokenized_dataset = drug_dataset.map(
    #    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names)
    
    # 2-  making the old columns the same size as the new ones.
    '''
    ->the tokenizer returns when we set return_overflowing_tokens=True
        overflow_to_sample_mapping field  
    ->It gives us a mapping from a new feature index to the index of the sample 
        it originated from. 
    ->Using this, we can associate each key present in our original dataset 
        with a list of values of the right size by repeating the values of each 
        example as many times as it generates new features:
    
    '''
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_and_split(examples):
        result = tokenizer(
            examples["review"],
            truncation=True,
            max_length=128,
            return_overflowing_tokens=True,
        )
        # input_id : list ,  attention_mask : list
        # examples keys : list , 
        # Extract mapping between new and old indices
        sample_map = result.pop("overflow_to_sample_mapping")
        for key, values in examples.items():
            result[key] = [values[i] for i in sample_map]
        return result
    
    samples_dataset = raw_ds['train'].select(range(2))

    samples_dataset = samples_dataset.map(tokenize_and_split,batched=True)


    # saving : parquet ( only for dataset) or arrow format (for datadict also)

    # Parquet
    parquet_path = Path(__file__).parent / 'parquet_data'
    parquet_path.mkdir(exist_ok=True)

    for split , dataset in raw_ds:
        dataset.to_parquet(f'{split}-data.parquet')
    # Load after
    ds_train = load_dataset('parquet',parquet_path/'train-data.parquet')

    # Arrow

    # save
    raw_ds.save_to_disk(data_dir) # saves all splits

    # load
    raw_ds.load_from_disk(data_dir) # saves all splits