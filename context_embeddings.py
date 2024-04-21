import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import time
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2TokenizerFast
from transformers import BertTokenizer, BertModel
import numpy as np
import pprint

nltk.data.path.append('./nltk_data')
wnl = WordNetLemmatizer()
bad_count = 0

def remove_non_alphanumeric(s):
    return ''.join([char for char in s if char.isalnum()])


def find_word_in_sentence(word, tokens):
    def full_lemmatize(word):
        word_types = ["n", "v", "a"]
        base_words = set()
        for c in word_types:
            base_words.add(wnl.lemmatize(word, c).lower())
        return base_words

    base_words = full_lemmatize(word)
    
    tokens = [token.lower() for token in tokens]
    tokens = [full_lemmatize(token) for token in tokens]

    for i, token_set in enumerate(tokens):
        if len(base_words.intersection(token_set)) != 0:
            return i
    return -1

def embed(word, context, model, tokenizer):
    inputs = tokenizer(context, return_tensors="pt")

    token_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [remove_non_alphanumeric(item) for item in tokens]
    clean_tokens = [wnl.lemmatize(remove_non_alphanumeric(item)) for item in tokens]
    
    word_index = find_word_in_sentence(word, clean_tokens)
    if word_index == -1:
        global bad_count
        bad_count += 1
        return np.zeros(10)

    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0, word_index]
    return embedding.detach().numpy().reshape(1, -1)

def embed_all_words_given_model(model_name, model, tokenizer):
    with open('./data/words/paragraphs.pkl', 'rb') as f:
        sentences = pickle.load(f)
    
    embeddings = {}
    count = 0
    start_time = time.time()
    for word in sentences:
        if count != 0 and count % 500 == 0:
            print(f"{count} words processed")
            print(f"{time.time() - start_time} time taken")
            print(f"No index count: {bad_count}")
            with open(f"./data/contextual_embeddings/{model_name}_word_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        sents = sentences[word]
        for i, sent in enumerate(sents):
            word_string = f"{word}_{i}"
            embedding = embed(word, sent, model, tokenizer)
            if np.all(embedding == 0):
                continue
            embeddings[word_string] = embedding
        count += 1

    with open(f"./data/contextual_embeddings/{model_name}_word_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
        

if __name__ == "__main__":
    configuration = DebertaV2Config()
    model = DebertaV2Model(configuration)
    tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v2-xlarge')

    embed_all_words_given_model("deberta", model, tokenizer)