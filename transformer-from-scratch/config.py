from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 10**-4
    seq_len: int = 350
    d_model: int = 512
    datasource: str = 'opus_books'
    lang_src: str = "en"
    lang_tgt: str = "fr"
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: str = "latest"
    tokenizer_file: str = "tokenizer_{0}.json"
    experiment_name: str = "runs/tmodel"

    def get_weights_file_path(self, epoch: str) -> str:
        model_folder = f"{self.datasource}_{self.model_folder}"
        model_filename = f"{self.model_basename}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

    def latest_weights_file_path(self) -> Optional[str]:
        model_folder = f"{self.datasource}_{self.model_folder}"
        model_filename = f"{self.model_basename}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])


if __name__=="__main__":
    config = Config()

    weights_file_path = config.get_weights_file_path("10")
    print("Specific weights file path:", weights_file_path)

    latest_weights = config.latest_weights_file_path()
    print("Latest weights file path:", latest_weights)
