from transformers import BertModel
from pathlib import Path

model_dir = Path(__file__).parent / 'model'

model = BertModel.from_pretrained(model_dir)

