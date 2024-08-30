from transformers import AutoTokenizer 
from datasets import load_dataset
from pathlib import Path


def get_text_gen(train_dataset):
    nb_sentences = 1000
    for i in range(0,len(train_dataset),nb_sentences):
        samples = train_dataset[i:i+nb_sentences]
        yield samples['whole_func_string']



if __name__=="__main__":
    raw_dataset= load_dataset("code_search_net", "python",trust_remote_code=True)

    old_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    corpus_source = get_text_gen(raw_dataset['train'])
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=corpus_source,
        vocab_size=52000,
    )

    example = """class LinearLayer():
        def __init__(self, input_size, output_size):
            self.weight = torch.randn(input_size, output_size)
            self.bias = torch.zeros(output_size)

        def __call__(self, x):
            return x @ self.weights + self.bias
        """
    print('old : ',old_tokenizer.tokenize(example))
    print('new : ',new_tokenizer.tokenize(example))


    tokenizer_folder = Path(__file__).parent / 'tokenizer'
    tokenizer_folder.mkdir(exist_ok=True,parents=True)


    new_tokenizer.save_pretrained(str(tokenizer_folder))
