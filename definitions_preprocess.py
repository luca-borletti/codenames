""" 
Embedding words' definitions for use by the spymaster bot.

Uses OpenAI's text-embedding-3-small model to embed words.
"""

from openai import OpenAI
import numpy as np
import pickle
import tqdm
import pprint

import time
import dotenv
import os
import requests
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WORDS_FILE_PATH = "./data/words/official_10000_english_words.txt"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


MAX_MAJOR_DEFS = 3
MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF = 2
MAX_MINOR_DEFS_PER_PART_OF_SPEECH = 1
MAX_DEFINITIONS = MAX_MAJOR_DEFS * MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF * MAX_MINOR_DEFS_PER_PART_OF_SPEECH


def get_definitions_merriam(word):
    # intermediate dictionary
    url = f"https://dictionaryapi.com/api/v3/references/sd3/json/{word}?key=0d985e9f-44be-4b21-aab4-fbcd68a58ad6" 
    
    # school dictionary
    url = f"https://dictionaryapi.com/api/v3/references/sd4/json/{word}?key=0caae7d2-19ff-4764-aed9-c65c1bbe6cdf"
    response = requests.get(url)
    try:
        # pprint.pprint(response.json())
        data = response.json()
        # filter out defs without 'word' or 'word:' in the ['meta']['id'] field
        major_definitions_groups = []
        for d in data:
            # if 'meta' in d and 'id' in d['meta'] and (word == ((d['meta']['id']).strip().lower()) or f'{word}:' in ((d['meta']['id']).strip().lower())):
            major_definitions_groups.append(d['shortdef'])
        major_definitions_groups = major_definitions_groups[:MAX_MAJOR_DEFS]
        
        count = 0
        result_definitions = []
        
        while count < MAX_DEFINITIONS and major_definitions_groups:
            new_major_definitions_groups = []
            for major_definitions in major_definitions_groups:
                add_defs = major_definitions[:MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF]
                result_definitions.extend(add_defs)
                count += len(add_defs)
                if count >= MAX_DEFINITIONS:
                    break
                defs_left = major_definitions[MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF:]
                if defs_left:
                    new_major_definitions_groups.append(defs_left)
            major_definitions_groups = new_major_definitions_groups
        
        return result_definitions
        
    except Exception as e:
        return None
    
    

def get_definitions_dictionaryapi(word):
    """
    Get the definitions of a word from the dictionaryapi.dev API.

    Args:
        word (string) - the word to get the definitions of

    Returns:
        definitions (list) - list of definitions of the word
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    try:
        definitions = []
        major_defs = response.json()
        for major_def in major_defs[:MAX_MAJOR_DEFS]:
            parts_of_speech = major_def["meanings"]
            for pos in parts_of_speech[:MAX_PARTS_OF_SPEECH_PER_MAJOR_DEF]:
                minor_defs = pos["definitions"]
                for minor_def in minor_defs[:MAX_MINOR_DEFS_PER_PART_OF_SPEECH]:
                    definition = minor_def["definition"]
                    definitions.append(definition)
                
        
        major_def = major_defs[0]
        parts_of_speech = major_def["meanings"]
        while len(definitions) < MAX_DEFINITIONS and parts_of_speech:
            # greedily add definitions from the first major definition
            pos = parts_of_speech[0]
            parts_of_speech = parts_of_speech[1:]
            minor_defs = pos["definitions"]
            while len(definitions) < MAX_DEFINITIONS and minor_defs:
                definition = minor_defs[0]["definition"]
                minor_defs = minor_defs[1:]
                if definition not in definitions:
                    definitions.append(definition)
                
        return definitions
    except Exception as e:
        return None

def embed_definitions(words):
    import time
    """
    Embeds a list of words into numpy vectors.

    Args:
        words (list) - list of words to embed
        pickle_path (string) - Stores the results in pickle_path

    Returns:
        embeddings (dict) - dictionary mapping each word to its vector embedding
    """
    print(f"Embedding definitions for {len(words)} words")

    word_to_def_embeddings = {}
    
    counts = {}

    try: 
        pickle_path = f"./data/embeddings/word_definitions.pkl"
        # read from pickle file if it exists
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                word_to_def_embeddings = pickle.load(f)
                num_words = len(word_to_def_embeddings)
                print(f"Loaded {num_words} word's embeddings from {pickle_path}")

        batch_size = 1000
        word_batches = []
        for i in range(0, len(words[num_words:]), batch_size):
            word_batches.append(words[num_words:][i:i+batch_size])

        for batch in tqdm.tqdm(word_batches):
            for word in tqdm.tqdm(batch):
                definitions = get_definitions_dictionaryapi(word)
                if definitions:
                    final_definitions = []
                    definition_embeddings = []
                    for definition in definitions:
                        try:
                            response = client.embeddings.create(model="text-embedding-3-small", input=[definition])
                            response = response.data[0].embedding
                            definition_embeddings.append(np.array(response))
                            final_definitions.append(definition)
                        except:
                            continue
                    definition_embeddings = np.array(definition_embeddings)
                    word_to_def_embeddings[word] = { "definitions": final_definitions, "embeddings": definition_embeddings }
                else:
                    word_to_def_embeddings[word] = { "definitions": [], "embeddings": [] }
                num_definitions = len(definitions) if definitions else 0
                counts[num_definitions] = counts.get(num_definitions, 0) + 1
                time.sleep(1)
            print(f"Finished embedding {len(word_to_def_embeddings)} words")
            print(f"Counts: {counts}")
            with open(pickle_path, "wb") as f:
                pickle.dump(word_to_def_embeddings, f)
                
    except KeyboardInterrupt:
        pass
    
    except Exception as _:
        pass

    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    print("\n".join([f"{k}: {v}" for k, v in sorted_counts]))
    
    return word_to_def_embeddings

def load_definitions():
    defs_pickle_file_path = "./data/words/dictionaryapi_word_defs.pkl"
    with open(defs_pickle_file_path, "rb") as f:
        word_to_def = pickle.load(f)
    return word_to_def

def load_definition_embeddings(model):
    embeddings_pickle_file_path = f"./data/definition_embeddings/{model}_word_defembeddings.pkl"
    with open(embeddings_pickle_file_path, "rb") as f:
        word_to_def_embeddings = pickle.load(f)
    return word_to_def_embeddings

if __name__ == '__main__':
    pass