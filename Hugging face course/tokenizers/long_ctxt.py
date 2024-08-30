from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

def example_on_overflowing_tokens(): 

    sentence = "This sentence is not too long but we are going to split it anyway."
    
    inputs = tokenizer(
        sentence, 
        truncation=True,
        return_overflowing_tokens=True,
        max_length=8, 
        stride=2
    )

    for ids in inputs["input_ids"]:
        print(tokenizer.decode(ids))

    print(inputs)

def example_on_overflow_sample_mapping():
    sentences = [
        "This sentence is not too long but we are going to split it anyway.",
        "This sentence is shorter but will still get split.",
    ]
    inputs = tokenizer(
        sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2,
    )

    for input in inputs["input_ids"]:
        print(tokenizer.decode(input))

        # print(inputs["input_ids"].shape) : (nb_encoding,max_length)
        # print(inputs['overflow_to_sample_mapping']) : (nb_encoding,)



question = "Which deep learning libraries backed by 🤗 Transformers?"

long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

inputs = tokenizer(
    question,
    long_context,
    stride=128, # overlap between each chunk
    max_length=384, 
    padding="longest", # pads to form tensors
    truncation="only_second", # do not trunc question
    return_overflowing_tokens=True, # return chunks
    return_offsets_mapping=True, # map each token to correpsing span of characters
)

_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping") # (nb_chunks , nb tokens)



inputs = inputs.convert_to_tensors("pt")
# print(inputs["input_ids"].shape) # (nb chunks , max length)

# long context was split in two, which means that after it goes through our model, 
# we will have two sets of start and end logits:

outputs = model(**inputs)

start_logits = outputs.start_logits # (nb chunks , max length)
end_logits = outputs.end_logits # (nb chunks , max length)

sequence_ids = inputs.sequence_ids(1) # (max_kength ,) :  this is for first chunk

# Mask everything apart from the tokens of the context

# mask the context
mask = [i != 1 for i in sequence_ids] # mask qst for 1st & 2nd chunk

# Unmask the [CLS] token
mask[0] = False

# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

# print(mask) (nb chunks , max length)
# print(inputs["attention_mask"]) : (nb_chunks , max_length)

start_logits[mask] = -10000
end_logits[mask] = -10000

start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

# get candidates from each chunk
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)