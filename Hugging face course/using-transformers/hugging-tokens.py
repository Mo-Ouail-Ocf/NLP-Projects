from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

sentence = 'Hello world'

# 1- genertae tokens
initial_tokens = tokenizer.tokenize(sentence)
print('Initial tokens :',initial_tokens)
# 2- Convert tokens to correponsing ids
input_tokens_ids = tokenizer.convert_tokens_to_ids(initial_tokens)
print('Input tokens initial :',input_tokens_ids)
# 3- Append special tokens ids 
input_tokens = tokenizer.prepare_for_model(input_tokens_ids)['input_ids']
print('Input tokens :',input_tokens)
# 4- Decode
decoded_ids = tokenizer.decode(input_tokens)
print('Decoded ids : ',decoded_ids)

# All in one
sentence = ['Hello world']
input_tokens = tokenizer(sentence) # 'token_type_ids' , 'attention_mask' 
print('All in one : ',input_tokens['input_ids'])
print(tokenizer.decode(input_tokens['input_ids'][0]))