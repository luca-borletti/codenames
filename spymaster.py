"""
Text embedding-based spymaster hint generation for Codenames.
 
ALL_WORDS - Assume we have some storage of all of the words.
# THRESHOLD - Assume we have a threshold for the worst cosine similarity between a 
# board word and the potential spy word.

DISTANCE(x, y) - Assume we have a function for cosine similarity between two embeddings.
SUBSETS(S) - Assume we have a function that returns all of the nonempty subsets of a set.

Initialize best subsets heap, BEST_SUBSETS, with size n
    Has elements (distance, subset)
Pull 24 random words, BOARD_WORDS, from ALL_WORDS
Put 12 words on the our team OUR_WORDS, and the rest are THEIR_WORDS
Iterate through all nonempty subsets, SUBSET, of OUR_WORDS:
    Iterate through all words, WORD, in ALL_WORDS:
        If WORD is not in BOARD_WORDS:
            BAD_DIST = minimum distance between the word and all words in THEIR_WORDS
            GOOD_DIST = maximum distance between the word and all words in SUBSET
            if BAD_DIST > GOOD_DIST and GOOD_DIST < BEST_DIST:
                Add (GOOD_DIST, SUBSET) to BEST_SUBSETS
                If the size of BEST_SUBSETS is greater than n, remove the smallest element
Return the subset with the smallest distance in BEST_SUBSETS
"""

import pprint
import time
import numpy as np
import heapq
from preprocess import load_embeddings
from guesser import guess_from_hint
from sklearn.metrics.pairwise import cosine_similarity
import sys
import tqdm
import pickle
from context_preprocess import load_context_embeddings

CODENAMES_WORDS_FILE_PATH = "./data/words/codenames_words.txt"

def load_codenames_words():
    with open(CODENAMES_WORDS_FILE_PATH, "r") as f:
        words = f.read().lower().splitlines()
    return words

CODENAMES_WORDS = load_codenames_words()

BOARD_SIZE = 16
OUR_SIZE = 8

DEBUG = False
def cosine_similarity2(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def similarity(x, y):
    return cosine_similarity2(x, y)

def jack_get_best_hint_of_same_size(words_to_embeddings, all_words, board_words, our_words, their_words, size = 2):
    best_hint = None
    best_sim = 0

    our_words_set = set(our_words)
    our_words_array = [words_to_embeddings[word] for word in our_words]
    our_words_array = np.vstack(our_words_array)

    their_words_array = [words_to_embeddings[word] for word in their_words]
    their_words_array = np.vstack(their_words_array)

    for word in all_words:
        if word in our_words_set:
            continue
        word_embedding = words_to_embeddings[word]
        our_similarities = cosine_similarity(our_words_array, [word_embedding]).reshape(-1)
        their_similarities = cosine_similarity(their_words_array, [word_embedding]).reshape(-1)

        indices = np.argpartition(our_similarities, -1 * size)[-1 * size:]
        index = indices[0]
        subset = [our_words[i] for i in indices]
        our_worst_sim = our_similarities[index]
        their_best_sim = np.max(their_similarities)

        if their_best_sim < our_worst_sim:
            if our_worst_sim > best_sim:
                best_sim = our_worst_sim
                best_hint = (subset, word)
    return best_hint
    
def jack_and_luca_get_best_hint_of_same_size_for_multidefs(batch_size, words_to_multi_embeddings, all_words, board_words, our_words, their_words, size = 2):
    '''
    words_to_multi_embeddings maps a word to a numpy vector of size n x m.
    n - the number of difference embeddings for the word
    m - dimension of embeddings
    '''
    best_hint = None
    best_sim = 0

    our_words_embeddings = [words_to_multi_embeddings[word] for word in our_words]
    our_words_embeddings = np.vstack(our_words_embeddings)
    our_word_index_to_embedding_indices = []
    curr_index = 0
    for i, word in enumerate(our_words):
        num_definitions = words_to_multi_embeddings[word].shape[0]
        our_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
        curr_index += num_definitions

    their_words_embeddings = [words_to_multi_embeddings[word] for word in their_words]
    their_words_embeddings = np.vstack(their_words_embeddings)
    their_word_index_to_embedding_indices = []
    curr_index = 0
    for i, word in enumerate(their_words):
        num_definitions = words_to_multi_embeddings[word].shape[0]
        their_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
        curr_index += num_definitions

    all_words = [word for word in all_words if word not in board_words]

    for i in range(0, len(all_words), batch_size):
        batch_words = all_words[i:i+batch_size]
        batch_embeddings = [words_to_multi_embeddings[word] for word in batch_words]
        batch_embeddings = np.vstack(batch_embeddings)
        batch_word_index_to_embedding_indices = []
        curr_index = 0
        for i, word in enumerate(batch_words):
            num_definitions = words_to_multi_embeddings[word].shape[0]
            batch_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
            curr_index += num_definitions
        
        our_similarities = cosine_similarity(our_words_embeddings, batch_embeddings)
        their_similarities = cosine_similarity(their_words_embeddings, batch_embeddings)
        
        max_similarities_per_our_word = np.array([np.max(our_similarities[indices], axis=0) for indices in our_word_index_to_embedding_indices])
        max_similarities_per_their_word = np.array([np.max(their_similarities[indices], axis=0) for indices in their_word_index_to_embedding_indices])
        
        max_our_similarities_per_batch_word = np.array([np.max(max_similarities_per_our_word[:, indices], axis=1) for indices in batch_word_index_to_embedding_indices])
        max_our_similarities_per_batch_word = np.transpose(max_our_similarities_per_batch_word)
        
        max_their_similarities_per_batch_word = np.array([np.max(max_similarities_per_their_word[:, indices], axis=1) for indices in batch_word_index_to_embedding_indices])
        max_their_similarities_per_batch_word = np.transpose(max_their_similarities_per_batch_word)
        
        our_best_sims_indices = np.argpartition(max_our_similarities_per_batch_word, -1 * size, axis=0)[-1 * size:]
        our_worst_sims_indices = our_best_sims_indices[0, :]
        our_worst_sims = max_our_similarities_per_batch_word[our_worst_sims_indices, range(max_our_similarities_per_batch_word.shape[1])]
        
        their_best_sims = np.max(max_their_similarities_per_batch_word, axis=0)
        
        for j in range(len(batch_words)):
            if their_best_sims[j] < our_worst_sims[j]:
                if our_worst_sims[j] > best_sim:
                    best_sim = our_worst_sims[j]
                    best_hint = ([our_words[i] for i in our_best_sims_indices[:, j]], batch_words[j])
    return best_hint

def initialize_game(all_words):
    board_words = np.random.choice(all_words, BOARD_SIZE, replace=False)
    our_words = np.random.choice(board_words, OUR_SIZE, replace=False)
    their_words = [word for word in board_words if word not in our_words]
    return list(board_words), list(our_words), list(their_words)

def make_games_context_embeddings(model):
    number_of_games = 500
    subset_size_to_evaluate = 3
    game_words = CODENAMES_WORDS
    words_to_embeddings = load_context_embeddings(model)

    for word in words_to_embeddings:
        words_to_embeddings[word] = np.vstack(words_to_embeddings[word])

    for word in game_words:
        if word not in words_to_embeddings:
            game_words.remove(word)

    dictionary_words = list(words_to_embeddings.keys())

    for _ in range(number_of_games):
        start = time.time()
        board_words, our_words, their_words = initialize_game(game_words)
        (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(500, words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size_to_evaluate)
        print(f"Hint: {best_hint_word}")
        print(f"Our words: {our_words}")
        print(f"Their words: {their_words}")
        print(f"Intended Guesses: {best_hint_subset}")
        print("\n\n")
        duration = time.time() - start
        print(f"Duration: {duration}")

def evaluate_spymaster_with_guesser_bot(model, type_of_embedding="MULTI-DIM"):
    number_of_games = 500
    subset_size_to_evaluate = 2
    game_words = CODENAMES_WORDS
    if type_of_embedding == "MULTI-DIM":
        words_to_embeddings = load_context_embeddings(model)
        for word in words_to_embeddings:
            words_to_embeddings[word] = np.vstack(words_to_embeddings[word])
    elif type_of_embedding == "NO MULTI-DIM":
        words_to_embeddings = load_embeddings(model)

    for word in game_words:
        if word not in words_to_embeddings:
            game_words.remove(word)

    dictionary_words = list(words_to_embeddings.keys())
    
    
    """ 
    Each game, we will initialize the game, get the best hints for each subset, and GPTGuesser guess the words.
    If GPTGuesser guessed the words correctly, we will add the number of correct guesses to the total correct guesses.
    We will also add the number of guesses to the total guesses.
    """
    
    total_duration = 0
    num_games = 0
    false_GPT_resp = 0
    intended_guesses = 0
    total_guesses = 0
    correct_guesses = 0
    perfect_hints = 0
    
    for _ in range(number_of_games):
        num_games += 1
        start = time.time()
        board_words, our_words, their_words = initialize_game(game_words)
        if type_of_embedding == "MULTI-DIM":
            (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(500, words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size_to_evaluate)
        elif type_of_embedding == "NO MULTI-DIM":
            (best_hint_subset, best_hint_word) = jack_get_best_hint_of_same_size(words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size_to_evaluate)
        
        guessed_words = guess_from_hint(board_words, best_hint_word, subset_size_to_evaluate)
        print(f"Hint: {best_hint_word}")
        print(f"Our words: {our_words}")
        print(f"Their words: {their_words}")
        print(f"Intended Guesses: {best_hint_subset}")
        print(f"Bot guesses: {guessed_words}")
        print("\n\n")
        duration = time.time() - start
        print(f"Duration: {duration}")
        total_duration += duration
        
        if len(guessed_words) != subset_size_to_evaluate:
            false_GPT_resp += 1
            continue
        
        intended_guesses_this_round = len(set(best_hint_subset).intersection(set(guessed_words)))
        intended_guesses += intended_guesses_this_round
        correct_guesses += len(set(guessed_words).intersection(set(our_words)))
        total_guesses += subset_size_to_evaluate
        if intended_guesses_this_round == subset_size_to_evaluate:
            perfect_hints += 1
    old_stdout = sys.stdout
    with open(f"./data/results/{model}_results.txt", "a") as f:
        sys.stdout = f
        print(f"Model name:", model)
        print(f"Average duration: {total_duration / num_games}")
        print(f"GPT Misfires: {false_GPT_resp}")
        print(f"Number of games played: {number_of_games}")
        print(f"Correct intended guesses: {intended_guesses}")
        print(f"Correct green guesses: {correct_guesses}")
        print(f"Number of perfect hints given: {perfect_hints}")
        print(f"Total Guesses: {total_guesses}")
        print("\n\n\n\n")
    sys.stdout = old_stdout
    
def check_cosine_similarity(word1, word2, WORDS_TO_EMBEDDINGS):
    return similarity(WORDS_TO_EMBEDDINGS[word1], WORDS_TO_EMBEDDINGS[word2])
        
if __name__ == "__main__":
    type_of_embeddings = ["MULTI-DIM", "NO MULTI-DIM"]

    type_of_embedding = "NO MULTI-DIM"
    models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    model = "fasttext"
   
    type_of_embedding = "MULTI-DIM"
    models = ["bert", "deberta"]
    model = "bert"
    
    evaluate_spymaster_with_guesser_bot(model, type_of_embedding)