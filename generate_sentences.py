from openai import OpenAI
import dotenv
import os
import re
import pickle
import pprint
import time
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
from fastpunct import FastPunct


dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORDS_FILE_PATH = "./data/words/official_10000_english_words.txt"
nltk.data.path.append('./nltk_data')
wnl = WordNetLemmatizer()
fastpunct = FastPunct()

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def move_files():
    with open('./data/words/paragraphs2.pkl', 'rb') as f:
        sentences = pickle.load(f)

    with open('./data/words/paragraphs.pkl', 'wb') as f:
        pickle.dump(sentences, f)

def show_counts():
    with open('./data/words/paragraphs.pkl', 'rb') as f:
        sentences = pickle.load(f)

    lengths = [len(sentences[key]) for key in sentences]
    print(Counter(lengths))

def strip_null_sentences():
    with open('./data/words/paragraphs2.pkl', 'rb') as f:
        sentences = pickle.load(f)
    
    for key in sentences:
        sents = sentences[key]
        remove = set()
        for i, sent in enumerate(sents):
            if sent.strip() == "":
                remove.add(i)
        if remove:
            sents = [sent for i, sent in enumerate(sents) if i not in remove]
            sentences[key] = sents
    with open('./data/words/paragraphs2.pkl', 'wb') as f:
        pickle.dump(sentences, f)

def delete_bad_keys():
    with open('./data/words/paragraphs2.pkl', 'rb') as f:
        sentences = pickle.load(f)
    remove_key = set()
    for key in sentences:
        sents = sentences[key]
        if len(sents) == 1:
            remove_key.add(key)
    for key in remove_key:
        sentences.pop(key)
    with open('./data/words/paragraphs2.pkl', 'wb') as f:
        pickle.dump(sentences, f)

def find_redo_words():
    with open('./data/words/paragraphs2.pkl', 'rb') as f:
        sentences = pickle.load(f)

    redo_words = set()
    for key in sentences:
        sents = sentences[key]
        if len(sents) != 3:
            redo_words.add(key)
    return redo_words

def remove_numbered_dot(text):
    pattern = r'^\d+\.\s*'
    result = re.sub(pattern, '', text)
    return result

def remove_non_alphanumeric(s):
    return ''.join([char for char in s if char.isalnum()])

def prompt_gpt3_turbo(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=200,
    )
    response_text = response.choices[0].message.content.lower()
    return response_text


def generate_sentences(word, version):
    if version == "paragraphs":
        prompt = f"Please list 3 multi-sentence example paragraphs for the word {word}," \
                "where each sentence uses a different definition." \
                f"Please use the exact word {word} and not some other version of the word."
    else:
        prompt = f"Please list 3 example sentences for the word {word}," \
                "where each sentence uses a different definition."
    result = prompt_gpt3_turbo(prompt)
    sentences = result.split("\n")
    sentences = list(map(remove_numbered_dot, sentences))
    return sentences

def generate_sentences_for_all_words(version, redo_words=None):
    with open(f'./data/words/{version}2.pkl', 'rb') as f:
        result = pickle.load(f)
    with open(WORDS_FILE_PATH, "r") as f:
        words = f.readlines()

    if redo_words == None:
        words = [word.lower().strip() for word in words]
    else:
        words = list(redo_words)

    start_time = time.time()
    for i, word in enumerate(words):
        if i != 0 and i % 10 == 0:
            print(f"{i} words done")
            print(f"{time.time() - start_time} seconds")
        lst = generate_sentences(word, version)
        result[word] = lst

    with open(f'./data/words/{version}2.pkl', 'wb') as f:
        pickle.dump(result, f)

def capitalize_sentences():
    with open(f'./data/words/paragraphs2.pkl', "rb") as f:
        paragraphs = pickle.load(f)
    start_time = time.time()
    for i, key in enumerate(paragraphs):
        if i != 0 and i % 100 == 0:
            print(f"{i}/{len(paragraphs)}")
            print(f"Time: {time.time() - start_time}")
        lst = paragraphs[key]
        new_lst = fastpunct.punct(lst)
        paragraphs[key] = new_lst
    
    with open(f'./data/words/paragraphs2.pkl', "wb") as f:
        pickle.dump(paragraphs, f)
            


if __name__ == "__main__":
    capitalize_sentences()



