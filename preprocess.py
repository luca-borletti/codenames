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
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WORDS_TO_EMBEDDINGS_FILE_PATH_1 = "./data/words/all_words1.pkl"
WORDS_TO_EMBEDDINGS_FILE_PATH_2 = "./data/words/all_words2.pkl"

GOOGLE_TOP_10000_WORDS = "./data/words/google-10000-english-usa-no-swears.txt"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def embed_words(words, pickle_path=False):
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
    
    for batch in word_batches:
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        response = response.data

        for i, word in enumerate(batch):
            embeddings[word] = np.array(response[i].embedding)

    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings

def load_embeddings(pickle_path):
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def load_all_embeddings():
    embeddings1 = load_embeddings(WORDS_TO_EMBEDDINGS_FILE_PATH_1)
    embeddings2 = load_embeddings(WORDS_TO_EMBEDDINGS_FILE_PATH_2)
    embeddings = {**embeddings1, **embeddings2}
    return embeddings

def check(pickle_path):
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    
    pprint.pprint(embeddings)
    pprint.pprint(len(embeddings))

def process_all_words():
    with open(GOOGLE_TOP_10000_WORDS, "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip()
    return words


if __name__ == '__main__':
    # words = process_all_words()
    # words = words[:5000]
    # embed_words(words, pickle_path=WORDS_TO_EMBEDDINGS_FILE_PATH_1)
    # check(WORDS_TO_EMBEDDINGS_FILE_PATH_1)

    # words = process_all_words()
    # words = words[5001:]
    # embed_words(words, pickle_path=WORDS_TO_EMBEDDINGS_FILE_PATH_2)
    # check(WORDS_TO_EMBEDDINGS_FILE_PATH_2)
    pass

