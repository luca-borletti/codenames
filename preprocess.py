from openai.types import CreateEmbeddingResponse, Embedding
from openai import OpenAI
import numpy as np
import pickle
import pprint

client = OpenAI(
    # This is the default and can be omitted
    api_key='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',
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
    

def check(pickle_path):
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    
    pprint.pprint(embeddings)
    pprint.pprint(len(embeddings))


def process_all_words():
    with open("google-10000-english-usa-no-swears.txt", "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip()
    return words


if __name__ == '__main__':
    words = process_all_words()
    words = words[:5000]
    embed_words(words, pickle_path="./all_words1.pkl")
    check("./all_words1.pkl")

    words = process_all_words()
    words = words[5001:]
    embed_words(words, pickle_path="./all_words2.pkl")
    check("./all_words2.pkl")