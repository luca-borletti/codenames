""" 
Embedding words for use by the spymaster bot.

Uses OpenAI's text-embedding-3-small model to embed words.
"""

from openai import OpenAI
import numpy as np
import pickle
import tqdm
import pprint

import dotenv
import os
import gensim.downloader
import nltk
from nltk.stem import WordNetLemmatizer
nltk.data.path.append('./nltk_data')

wnl = WordNetLemmatizer()

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WORDS_FILE_PATH = "./data/words/official_10000_english_words.txt"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def embed_words(words, model_name):
    """
    Embeds a list of words into numpy vectors.

    Args:
        words (list) - list of words to embed
        pickle_path (string) - Stores the results in pickle_path

    Returns:
        embeddings (dict) - dictionary mapping each word to its vector embedding
    """
    batch_size = 1000
    word_batches = []
    for i in range(0, len(words), batch_size):
        word_batches.append(words[i:i+batch_size])

    embeddings = {}
    
    
    if model_name == "openai":
        for batch in word_batches:
            response = client.embeddings.create(model="text-embedding-3-small", input=batch)
            response = response.data

            for i, word in enumerate(batch):
                embeddings[word] = np.array(response[i].embedding)
            print(f"Finished embedding {len(embeddings)} words")
    elif model_name == "glove300" or model_name == "word2vec300" or \
         model_name == "glove100" or model_name == "glovetwitter200" or model_name == "fasttext":
        if model_name == "glove300":
            model = gensim.downloader.load('glove-wiki-gigaword-300')
        elif model_name == "word2vec300":
            model = gensim.downloader.load('word2vec-google-news-300')
        elif model_name == "glove100":
            model = gensim.downloader.load('glove-wiki-gigaword-100')
        elif model_name == "glovetwitter200":
            model = gensim.downloader.load('glove-twitter-200')
        elif model_name == "fasttext":
            model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
        
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model:
                    embeddings[word] = np.array(model[word])
            print(f"Finished embedding {len(embeddings)} words")
    elif model_name == "word2vec+glove300":
        model1 = gensim.downloader.load('word2vec-google-news-300')
        model2 = gensim.downloader.load('glove-wiki-gigaword-300')
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model1 and word in model2:
                    vec1 = np.array(model1[word])
                    vec2 = np.array(model2[word])
                    embeddings[word] = np.concatenate((vec1, vec2))
            print(f"Finished embedding {len(embeddings)} words")

    pickle_path = f"./data/embeddings/{model_name}_word_embeddings.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

def load_embeddings(model_name):
    pickle_path = f"./data/embeddings/{model_name}_word_embeddings.pkl"
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def process_all_words():
    with open(WORDS_FILE_PATH, "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip().lower()
    return words

def lemmatize_words():
    words = process_all_words()
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(wnl.lemmatize(word))
    different_words = []
    for i, word in enumerate(words):
        if word != lemmatized_words[i]:
            different_words.append((word, lemmatized_words[i]))
    return different_words

if __name__ == '__main__':
    pprint.pprint(lemmatize_words())
    pass