from pathlib import Path
from transformers import BertConfig , BertModel

config = BertConfig()

model_dir = Path(__file__).parent / 'model'

# un-init model 
uninit_model = BertModel(config)

model = BertModel.from_pretrained('bert-base-cased')

model.save_pretrained(save_directory=model_dir)


