from openai import OpenAI
import dotenv
import os
import re
import pickle
import pprint
import time
from transformers import BertTokenizer, BertModel
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORDS_FILE_PATH = "./data/words/official_10000_english_words.txt"
nltk.data.path.append('./nltk_data')
wnl = WordNetLemmatizer()

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

def embed(word, context, model, tokenizer):
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model(**inputs)

    token_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    clean_tokens = [wnl.lemmatize(remove_non_alphanumeric(item)) for item in tokens]
    
    word_index = clean_tokens.index(word)

    embedding = outputs.last_hidden_state[0, word_index]
    return embedding.detach().numpy().reshape(1, -1)

def embed_all_words_given_model(model_name, model, tokenizer):
    with open('./data/words/sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)
    
    embeddings = {}
    count = 0
    start_time = time.time()
    for word in sentences:
        if count != 0 and count % 10 == 0:
            print(f"{count} words processed")
            print(f"{time.time() - start_time} time taken")

        sents = sentences[word]
        for i, sent in enumerate(sents):
            word_string = f"{word}_{i}"
            embedding = embed(word, sent, model, tokenizer)
            embeddings[word_string] = embedding
        count += 1

    with open(f"./data/contextual_embeddings/{model_name}_word_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def embed_all_words(model_name):
    if model_name == "bert_base":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    else:
        return

    embed_all_words_given_model(model_name, model, tokenizer)


def find_word_in_sentence(word, sentence):
    def full_lemmatize(word):
        word_types = ["n", "v", "a"]
        base_words = set()
        for c in word_types:
            base_words.add(wnl.lemmatize(word, c).lower())
        return base_words

    base_words = full_lemmatize(word)
    
    tokens = nltk.word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens = [full_lemmatize(token) for token in tokens]

    for i, token_set in enumerate(tokens):
        if len(base_words.intersection(token_set)) != 0:
            return i
    return -1

    
    
            


if __name__ == "__main__":
    pass



