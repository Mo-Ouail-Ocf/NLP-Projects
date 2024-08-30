from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

sentence = ['Hello world , I am small intesting sentence',
            'Hello world , I am small sentence']

# tokenizer keeps track of
# 1- words each token come from
# 2- span of words each token come from w/ return_offsets_mapping=True parameter 

encoding=tokenizer(sentence,return_offsets_mapping=True)
print(encoding.tokens(1))

print(encoding.tokens()) # resulting tokens for 1st sentence
print(encoding.word_ids()) # = input_ids
print(encoding['offset_mapping']) # maps each token to the correpsonding span of chars 
print(encoding.sequence_ids()) # = token_types_ids """ """
# Internal pipeline :
# 1-Normalization
# 2-Pre-Tokenization : split words into tokens
# 3- Aplly model
# 4- special tokens adding
# tokenizer keeps span of each word & each token

print(encoding.token_to_chars(0)) # token index -> CharSpan ( start , end )
print(encoding.word_to_chars(0)) # word in sentence-> CharSpan ( start , end )
print(encoding.char_to_token(0)) # char position in sentence -> token index
print(encoding.char_to_word(0)) # char position in sentence -> word span 

print('Normalization : ',tokenizer.backend_tokenizer.normalizer)