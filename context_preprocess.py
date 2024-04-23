import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import time
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2TokenizerFast
from transformers import BertTokenizer, BertModel
import numpy as np
import pprint
from collections import Counter
import torch
from sklearn.metrics.pairwise import cosine_similarity

nltk.data.path.append('./nltk_data')
wnl = WordNetLemmatizer()
bad_count = 0

def load_context_embeddings(model_name):
    pickle_path = f"data/contextual_embeddings/{model_name}_word_embeddings.pkl"
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

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

    return_set = set()
    for i, token_set in enumerate(tokens):
        if len(base_words.intersection(token_set)) != 0:
            return_set.add(i)
    return return_set

def embed_singular_word(word, context, model, tokenizer):
    inputs = tokenizer(context, return_tensors="pt")

    token_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [remove_non_alphanumeric(item) for item in tokens]
    clean_tokens = [wnl.lemmatize(remove_non_alphanumeric(item)) for item in tokens]
    
    word_index_set = find_word_in_sentence(word, clean_tokens)
    if len(word_index_set) == 0:
        global bad_count
        bad_count += 1
        return np.zeros(10)

    outputs = model(**inputs)
    embeddings = []
    for word_index in word_index_set:
        embedding = outputs.last_hidden_state[0, word_index]
        embeddings.append(embedding)
    stacked_tensors = torch.stack(embeddings)
    average_tensor = torch.mean(stacked_tensors, dim=0)
    return average_tensor.detach().numpy().reshape(1, -1)

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
            embedding = embed_singular_word(word, sent, model, tokenizer)
            if np.all(embedding == 0):
                continue
            if word not in embeddings:
                embeddings[word] = []
            embeddings[word].append(embedding)
        count += 1

    with open(f"./data/contextual_embeddings/{model_name}_word_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def embed(model_name):
    if model_name == "deberta":
        configuration = DebertaV2Config()
        model = DebertaV2Model(configuration)
        tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v2-xlarge')
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")    

    embed_all_words_given_model(model_name, model, tokenizer)
    
def check_board_words(model_name):
    '''
    Checks how many board words are properly embedded
    '''
    with open("./data/words/codenames_words.txt", "r") as f:
        words = f.read().lower().splitlines()
    words = set(words)
    print("All codename words:", len(words))

    with open(f"./data/contextual_embeddings/cleaned_{model_name}_word_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    embeddings_keys = set(embeddings.keys())
    print("Embedded words:", len(embeddings_keys))
    print("Intersection:", len(words.intersection(embeddings_keys)))

def find_distance(word1, word2, model_name):
    '''
    Finds distance between two words
    '''
    pickle_path = f"data/contextual_embeddings/cleaned_{model_name}_word_embeddings.pkl"
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    
    best_similarity = 0
    for i in range(3):
        for j in range(3):
            embedding1 = embeddings[word1][i]
            embedding2 = embeddings[word2][j]
            sim = cosine_similarity(embedding1, embedding2)
            sim = sim[0][0]
            best_similarity = max(sim, best_similarity)
    return best_similarity

def print_paragraphs(word):
    with open("data/words/paragraphs.pkl", "rb") as f:
        paragraphs = pickle.load(f)
    print(paragraphs[word])



if __name__ == "__main__":
    model_names = ["deberta", "bert"]
    model_name = "bert"
    embed(model_name)


# Hint: square
# Our words: ['cat', 'march', 'center', 'lawyer', 'antarctica', 'string', 'circle', 'bottle']
# Their words: ['club', 'date', 'root', 'bat', 'conductor', 'olympus', 'amazon', 'nail']
# Intended Guesses: ['center', 'circle', 'march']
    
# Paragraphs for square:
# ['i saw a group of friends laughing and chatting in the town square, surrounded by colorful buildings and street vendors selling their goods. the square was bustling with activity as people milled about, enjoying the sunny day and lively atmosphere.', 'the answer to the math problem required me to find the square of the number 5, which turned out to be 25. after calculating the square of a few more numbers, i began to grasp the relationship between a number and its square.', 'the teacher instructed us to draw a square on the graph paper, making sure all four sides were of equal length and all four angles were right angles. as i carefully plotted each point and connected the lines, a perfect square took shape on the page.']