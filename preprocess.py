""" 
Embedding words for use by the spymaster bot.

Uses OpenAI's text-embedding-3-small model to embed words.
"""

from openai import OpenAI
import numpy as np
import pickle
import pprint

import dotenv
import os
import gensim.downloader
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
    words = [word.lower() for word in words]
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
    elif model_name == "glove":
        model = gensim.downloader.load('glove-wiki-gigaword-300')
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model.wv:
                    embeddings[word] = np.array(model.wv[word])
    elif model_name == "word2vec":
        model = gensim.downloader.load('word2vec-google-news-300')
        for batch in word_batches:
            for i, word in enumerate(batch):
                if word in model.wv:
                    embeddings[word] = np.array(model.wv[word])

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
        words[i] = words[i].strip()
    return words


if __name__ == '__main__':
    words = process_all_words()
    models = ["openai", "word2vec", "glove"]
    for model in models:
        embed_words(words, model)

