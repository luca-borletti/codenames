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

MAX_MEANINGS = 2
MAX_DEFINITIONS_PER_MEANING = 5

def get_definitions(word):
    """
    Get the definitions of a word from the dictionaryapi.dev API.

    Args:
        word (string) - the word to get the definitions of

    Returns:
        definitions (list) - list of definitions of the word
    """
    import requests
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    data = response.json()
    try:
        meanings = data[0]["meanings"]
        all_definitions = []
        for meaning in meanings[:MAX_MEANINGS]:
            curr_definitions = meaning["definitions"]
            for definition in curr_definitions[:MAX_DEFINITIONS_PER_MEANING]:
                try:
                    example = definition["example"] if "example" in definition else ""
                    definition_text = definition["definition"] + ((" " + example) if example else "")
                    all_definitions.append(definition_text)
                except:
                    continue
                    
        return all_definitions
    
    except:
        return None



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

def embed_definitions(words):
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
        for i, word in enumerate(batch):
            definitions = get_definitions(word)
            if definitions:
                definitions_and_embeddings = []
                for definition in definitions:
                    response = client.embeddings.create(model="text-embedding-3-small", input=[definition])
                    response = response.data[0].embedding
                    definitions_and_embeddings.append((definition, np.array(response)))
                embeddings[word] = definitions_and_embeddings
        print(f"Finished embedding {len(embeddings)} words")
    pickle_path = f"./data/embeddings/word_definitions.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(embeddings, f)
        
    return embeddings


if __name__ == '__main__':
    # words = process_all_words()
    # models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    # model = "fasttext"
    # embed_words(words, model)
    print(get_definitions("apple"))

