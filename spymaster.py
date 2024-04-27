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
from preprocess import load_embeddings, process_all_words
from guesser import guess_from_hint
from sklearn.metrics.pairwise import cosine_similarity
import sys
import tqdm
import pickle
from context_preprocess import load_context_embeddings
from definitions_preprocess import load_definition_embeddings
from collections import OrderedDict
import random

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
    best_hint = ""
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


def get_context_embedding(bert_embeddings, batch_words, our_words, their_words):
    _, bert_size_embedding = bert_embeddings[next(iter(bert_embeddings))].shape

    our_words_embeddings = [bert_embeddings[word] for word in our_words]
    our_words_embeddings = np.vstack(our_words_embeddings)
    our_word_index_to_embedding_indices = []
    curr_index = 0
    for i, word in enumerate(our_words):
        num_definitions = bert_embeddings[word].shape[0]
        our_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
        curr_index += num_definitions

    their_words_embeddings = [bert_embeddings[word] for word in their_words]
    their_words_embeddings = np.vstack(their_words_embeddings)
    their_word_index_to_embedding_indices = []
    curr_index = 0
    for i, word in enumerate(their_words):
        num_definitions = bert_embeddings[word].shape[0]
        their_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
        curr_index += num_definitions

    for word in batch_words:
        if word not in bert_embeddings:
            bert_embeddings[word] = np.zeros((1, bert_size_embedding))

    batch_embeddings = [bert_embeddings[word] for word in batch_words]
    batch_embeddings = np.vstack(batch_embeddings)
    batch_word_index_to_embedding_indices = []
    curr_index = 0
    for j, word in enumerate(batch_words):
        num_definitions = bert_embeddings[word].shape[0]
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

    return max_our_similarities_per_batch_word, max_their_similarities_per_batch_word

def jack_and_luca_get_best_hint_of_same_size_for_multidefs(words_to_multi_embeddings, all_words, board_words, our_words, their_words, bert_embeddings=None, bert_weight=0.2, size = 2):
    '''
    words_to_multi_embeddings maps a word to a numpy vector of size n x m.
    n - the number of difference embeddings for the word
    m - dimension of embeddings

    Same with bert_embeddings
    '''
    best_hint = ""
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

    batch_size = 500    
    all_words = [word for word in all_words if word not in board_words]

    for i in range(0, len(all_words), batch_size):
        batch_words = all_words[i:i+batch_size]
        batch_embeddings = [words_to_multi_embeddings[word] for word in batch_words]
        batch_embeddings = np.vstack(batch_embeddings)
        batch_word_index_to_embedding_indices = []
        curr_index = 0
        for j, word in enumerate(batch_words):
            num_definitions = words_to_multi_embeddings[word].shape[0]
            batch_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
            curr_index += num_definitions
            # maybe also construct a embedding_indices_to_word_index dictionary for use below

        our_similarities = cosine_similarity(our_words_embeddings, batch_embeddings)
        their_similarities = cosine_similarity(their_words_embeddings, batch_embeddings)

        # maybe use argmax, get indices of the rows that have the max similarity along each row
        # then for each element at col m, row i, with argmax j get the jth value of the ith element in the dict above to get the index of the row in the 
        # similarity matrix, and use same col m to get the similarity value
        max_similarities_per_our_word = np.array([np.max(our_similarities[indices], axis=0) for indices in our_word_index_to_embedding_indices]) 
        max_similarities_per_their_word = np.array([np.max(their_similarities[indices], axis=0) for indices in their_word_index_to_embedding_indices])


        max_our_similarities_per_batch_word = np.array([np.max(max_similarities_per_our_word[:, indices], axis=1) for indices in batch_word_index_to_embedding_indices])
        max_our_similarities_per_batch_word = np.transpose(max_our_similarities_per_batch_word)

        max_their_similarities_per_batch_word = np.array([np.max(max_similarities_per_their_word[:, indices], axis=1) for indices in batch_word_index_to_embedding_indices])
        max_their_similarities_per_batch_word = np.transpose(max_their_similarities_per_batch_word)
        
        if bert_embeddings:
            our_similarities_bert, their_similarities_bert = get_context_embedding(bert_embeddings, batch_words, our_words, their_words)
            max_their_similarities_per_batch_word += bert_weight * our_similarities_bert
            max_their_similarities_per_batch_word += bert_weight * their_similarities_bert

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



def weighted_spymaster_get_hint(models_to_words_to_embeddings, models_to_weights, dictionary_words, board_words, our_words, their_words, subset_size_to_evaluate):
    best_hint = ""
    best_sim = 0

    models_to_matrices = {model : {} for model in models_to_weights}

    for model, words_to_embeddings in models_to_words_to_embeddings.items():
        our_words_embeddings = [words_to_embeddings[word] for word in our_words]
        our_words_embeddings = np.vstack(our_words_embeddings)
        models_to_matrices[model]["our_words_embeddings"] = our_words_embeddings
        our_word_index_to_embedding_indices = []
        curr_index = 0
        for i, word in enumerate(our_words):
            num_definitions = words_to_embeddings[word].shape[0]
            our_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
            curr_index += num_definitions
        models_to_matrices[model]["our_word_index_to_embedding_indices"] = our_word_index_to_embedding_indices

        their_words_embeddings = [words_to_embeddings[word] for word in their_words]
        their_words_embeddings = np.vstack(their_words_embeddings)
        models_to_matrices[model]["their_words_embeddings"] = their_words_embeddings
        their_word_index_to_embedding_indices = []
        curr_index = 0
        for i, word in enumerate(their_words):
            num_definitions = words_to_embeddings[word].shape[0]
            their_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
            curr_index += num_definitions
        models_to_matrices[model]["their_word_index_to_embedding_indices"] = their_word_index_to_embedding_indices

    batch_size = 500    
    potential_hint_words = [word for word in dictionary_words if word not in board_words]

    for i in range(0, len(potential_hint_words), batch_size):
        batch_words = potential_hint_words[i:i+batch_size]
        for model, words_to_embeddings in models_to_words_to_embeddings.items():
            batch_embeddings = [words_to_embeddings[word] for word in batch_words]
            batch_embeddings = np.vstack(batch_embeddings)
            models_to_matrices[model]["batch_embeddings"] = batch_embeddings
            batch_word_index_to_embedding_indices = []
            curr_index = 0
            for j, word in enumerate(batch_words):
                num_definitions = words_to_embeddings[word].shape[0]
                batch_word_index_to_embedding_indices.append(list(range(curr_index, curr_index + num_definitions)))
                curr_index += num_definitions
                # maybe also construct a embedding_indices_to_word_index dictionary for use below
            models_to_matrices[model]["batch_word_index_to_embedding_indices"] = batch_word_index_to_embedding_indices

        max_our_similarities_per_batch_word = None
        max_their_similarities_per_batch_word = None

        for model, matrices in models_to_matrices.items():
            curr_our_similarities = cosine_similarity(matrices["our_words_embeddings"], matrices["batch_embeddings"])
            curr_their_similarities = cosine_similarity(matrices["their_words_embeddings"], matrices["batch_embeddings"])
            # print(curr_our_similarities.shape)
            
            curr_max_similarities_per_our_word = np.array([np.max(curr_our_similarities[indices], axis=0) for indices in matrices["our_word_index_to_embedding_indices"]]) 
            curr_max_similarities_per_their_word = np.array([np.max(curr_their_similarities[indices], axis=0) for indices in matrices["their_word_index_to_embedding_indices"]])
            # print(curr_max_similarities_per_our_word.shape)
            
            curr_max_our_similarities_per_batch_word = np.array([np.max(curr_max_similarities_per_our_word[:, indices], axis=1) for indices in matrices["batch_word_index_to_embedding_indices"]])
            curr_max_our_similarities_per_batch_word = np.transpose(curr_max_our_similarities_per_batch_word)
            # print(curr_max_our_similarities_per_batch_word.shape)

            curr_max_their_similarities_per_batch_word = np.array([np.max(curr_max_similarities_per_their_word[:, indices], axis=1) for indices in matrices["batch_word_index_to_embedding_indices"]])
            curr_max_their_similarities_per_batch_word = np.transpose(curr_max_their_similarities_per_batch_word)
            # print(curr_max_their_similarities_per_batch_word.shape)

            weight = models_to_weights[model]
            
            if max_our_similarities_per_batch_word is None:
                max_our_similarities_per_batch_word = weight * curr_max_our_similarities_per_batch_word
                max_their_similarities_per_batch_word = weight * curr_max_their_similarities_per_batch_word
            else:
                max_our_similarities_per_batch_word += weight * curr_max_our_similarities_per_batch_word
                max_their_similarities_per_batch_word += weight * curr_max_their_similarities_per_batch_word
            
        our_best_sims_indices = np.argpartition(max_our_similarities_per_batch_word, -1 * subset_size_to_evaluate, axis=0)[-1 * subset_size_to_evaluate:]
        our_worst_sims_indices = our_best_sims_indices[0, :]
        our_worst_sims = max_our_similarities_per_batch_word[our_worst_sims_indices, range(max_our_similarities_per_batch_word.shape[1])]
        
        their_best_sims = np.max(max_their_similarities_per_batch_word, axis=0)
        
        for j in range(len(batch_words)):
            if their_best_sims[j] < our_worst_sims[j]:
                if our_worst_sims[j] > best_sim:
                    best_sim = our_worst_sims[j]
                    best_hint = ([our_words[i] for i in our_best_sims_indices[:, j]], batch_words[j])
    return best_hint

def evaluate_generalized_weighted_spymaster_with_guesser_bot(models_and_types_to_weights, subset_size_to_evaluate):
    game_words = CODENAMES_WORDS
    dictionary_words = process_all_words()
    # print(f"Game words: {len(game_words)}")
    # print(f"Dictionary words: {len(dictionary_words)}")
    models_to_weights = {model : weights for (model, _), weights in models_and_types_to_weights.items()}
    models_and_types = list(models_and_types_to_weights.keys())
    models_to_words_to_embeddings = {}
    for model, type_of_embedding in models_and_types:
        if type_of_embedding == "MULTI-DIM-DEFINITION":
            words_to_embeddings = load_definition_embeddings(model)
        elif type_of_embedding == "MULTI-DIM-CONTEXT":
            words_to_embeddings = load_context_embeddings(model)
            for word in words_to_embeddings:
                words_to_embeddings[word] = np.vstack(words_to_embeddings[word])
        elif type_of_embedding == "NO MULTI-DIM":
            words_to_embeddings = load_embeddings(model)
            for word in words_to_embeddings:
                words_to_embeddings[word] = np.array(words_to_embeddings[word]).reshape(1, -1)
        models_to_words_to_embeddings[model] = words_to_embeddings
        curr_dictionary_words = list(words_to_embeddings.keys())
        if dictionary_words is None:
            dictionary_words = curr_dictionary_words
        else:
            dictionary_words = list((set(dictionary_words)).intersection(set(curr_dictionary_words)))
        game_words = list((set(game_words)).intersection(set(curr_dictionary_words))) # !!! PROBLEM the overlap SUCKS !!!
    # print(f"Dictionary words: {len(dictionary_words)}")
    # print(f"Game words: {len(game_words)}")
    
    number_of_games = 500
    
    total_duration = 0
    false_GPT_resp = 0
    intended_guesses = 0
    total_guesses = 0
    correct_guesses = 0
    perfect_hints = 0

    for i in range(number_of_games):
        start = time.time()
        board_words, our_words, their_words = initialize_game(game_words)
        (best_hint_subset, best_hint_word) = weighted_spymaster_get_hint(models_to_words_to_embeddings, models_to_weights, dictionary_words, board_words, our_words, their_words, subset_size_to_evaluate)
        
        guessed_words = guess_from_hint(board_words, best_hint_word, subset_size_to_evaluate)
        print(f"Board number: {i}")
        print(f"Hint: {best_hint_word}")
        print(f"Intended Guesses: {best_hint_subset}")
        print(f"Our words: {our_words}")
        print(f"Their words: {their_words}")
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
    models_and_weights_string = "_".join([f"{model}_{weights}" for (model, _), weights in models_and_types_to_weights.items()])
    run_name = f"./data/results/{models_and_weights_string}_subset_{subset_size_to_evaluate}_results.txt"
    with open(run_name, "w") as f:
        sys.stdout = f
        print(f"Model name:", model)
        print(f"Average duration: {total_duration / number_of_games}")
        print(f"GPT Misfires: {false_GPT_resp}")
        print(f"Number of games played: {number_of_games}")
        print(f"Correct intended guesses: {intended_guesses}")
        print(f"Correct green guesses: {correct_guesses}")
        print(f"Number of perfect hints given: {perfect_hints}")
        print(f"Total Guesses: {total_guesses}")
        print(f"Intended guesses percentage: {intended_guesses / total_guesses}")
        print(f"Green guesses percentage: {correct_guesses / total_guesses}")
        print("\n\n\n\n")
    sys.stdout = old_stdout


def evaluate_spymaster_with_guesser_bot(model, subset_size, use_bert_embeddings=True, bert_weight=0.2, type_of_embedding="MULTI-DIM-CONTEXT"):
    number_of_games = 500
    subset_size_to_evaluate = subset_size
    game_words = CODENAMES_WORDS
    if type_of_embedding == "MULTI-DIM-DEFINITION":
        words_to_embeddings = load_definition_embeddings(model)
    elif type_of_embedding == "MULTI-DIM-CONTEXT":
        words_to_embeddings = load_context_embeddings(model)
        for word in words_to_embeddings:
            words_to_embeddings[word] = np.vstack(words_to_embeddings[word])
    elif type_of_embedding == "NO MULTI-DIM":
        words_to_embeddings = load_embeddings(model)
        for word in words_to_embeddings:
            words_to_embeddings[word] = np.array(words_to_embeddings[word]).reshape(1, -1)
        
        if use_bert_embeddings:
            # bert_embeddings = load_context_embeddings("bert")
            bert_embeddings = load_definition_embeddings("openai")
            for word in bert_embeddings:
                bert_embeddings[word] = np.vstack(bert_embeddings[word])
        else:
            bert_embeddings = None

    dictionary_words = list(words_to_embeddings.keys())
    game_words = list((set(game_words)).intersection(set(dictionary_words)))

    if use_bert_embeddings:
        game_words = list((set(game_words)).intersection(set(bert_embeddings.keys())))
    
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
        if type_of_embedding == "MULTI-DIM-CONTEXT" or type_of_embedding == "MULTI-DIM-DEFINITION":
            (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size_to_evaluate)
        elif type_of_embedding == "NO MULTI-DIM":
            (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(words_to_embeddings, dictionary_words, board_words, our_words, their_words, bert_embeddings=bert_embeddings, bert_weight=bert_weight, size=subset_size_to_evaluate)
        
        guessed_words = guess_from_hint(board_words, best_hint_word, subset_size_to_evaluate)
        print(f"Board number: {num_games}")
        print(f"Hint: {best_hint_word}")
        print(f"Intended Guesses: {best_hint_subset}")
        print(f"Our words: {our_words}")
        print(f"Their words: {their_words}")
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
    with open(f"./data/results/{model}_{subset_size}_useOpenaiDefs:{use_bert_embeddings}{bert_weight}_results.txt", "w") as f:
        sys.stdout = f
        print(f"Model name:", model)
        print(f"Average duration: {total_duration / num_games}")
        print(f"GPT Misfires: {false_GPT_resp}")
        print(f"Number of games played: {number_of_games}")
        print(f"Correct intended guesses: {intended_guesses}")
        print(f"Correct green guesses: {correct_guesses}")
        print(f"Number of perfect hints given: {perfect_hints}")
        print(f"Total Guesses: {total_guesses}")
        print(f"Intended guesses percentage: {intended_guesses / total_guesses}")
        print(f"Green guesses percentage: {correct_guesses / total_guesses}")
        print("\n\n\n\n")
    sys.stdout = old_stdout

def find_multi_distance(word1, word2, embeddings):
    '''
    Finds distance between two words with a multidimensional embedding
    '''
    best_similarity = 0
    for embedding1 in embeddings[word1]:
        for embedding2 in embeddings[word2]:
            embedding1 = embeddings[word1]
            embedding2 = embeddings[word2]
            sim = cosine_similarity(embedding1, embedding2)
            sim = sim[0][0]
            best_similarity = max(sim, best_similarity)
    return best_similarity
    
def check_cosine_similarity(word1, word2, WORDS_TO_EMBEDDINGS):
    return similarity(WORDS_TO_EMBEDDINGS[word1], WORDS_TO_EMBEDDINGS[word2])

def play_one_game_on_single_model(model, subset_size, type_of_embedding, our_words, their_words):
    board_words = our_words + their_words
    if type_of_embedding == "MULTI-DIM-DEFINITION":
        words_to_embeddings = load_definition_embeddings(model)
    elif type_of_embedding == "MULTI-DIM-CONTEXT":
        words_to_embeddings = load_context_embeddings(model)
        for word in words_to_embeddings:
            words_to_embeddings[word] = np.vstack(words_to_embeddings[word])
    elif type_of_embedding == "NO MULTI-DIM":
        words_to_embeddings = load_embeddings(model)
        for word in words_to_embeddings:
            words_to_embeddings[word] = np.array(words_to_embeddings[word]).reshape(1, -1)
    
    dictionary_words = list(words_to_embeddings.keys())
    
    (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size)
    
    return best_hint_subset, best_hint_word

def test_game_on_all_models():
    our_words = ['princess', 'deck', 'shop', 'plane', 'leprechaun', 'glass', 'crane', 'root'],
    their_words = ['horn', 'bank', 'limousine', 'pistol', 'court', 'hand', 'ivory', 'dwarf']
    for model, type_of_embedding in [("openai", "MULTI-DIM-DEFINITION"), ("word2vec300", "NO MULTI-DIM"), ("glove300", "NO MULTI-DIM"), ("word2vec+glove300", "NO MULTI-DIM"), ("glove100", "NO MULTI-DIM"), ("glovetwitter200", "NO MULTI-DIM"), ("fasttext", "NO MULTI-DIM")]:
        print(f"Model: {model}")
        print(f"Type of embedding: {type_of_embedding}")
        best_hint_subset, best_hint_word = play_one_game_on_single_model(model, 3, type_of_embedding, our_words, their_words)
        print(f"Hint: {best_hint_word}")
        print(f"Intended Guesses: {best_hint_subset}")
        print("\n\n")

def experts_framework():
    models_to_weights = OrderedDict({
        ("bert", "MULTI-DIM-CONTEXT") : [18], 
        ("openai", "MULTI-DIM-DEFINITION") : [23], 
        ("word2vec300", "NO MULTI-DIM") : [22], 
        ("glove300", "NO MULTI-DIM") : [18], 
        ("fasttext", "NO MULTI-DIM") : [19]
    })
    
    models = list(models_to_weights.keys())
    
    subset_sizes = [2, 3]
    
    number_of_games = 500
    
    total_load_time = 0
    total_time = 0
    false_GPT_resp = 0
    intended_guesses = 0
    total_guesses = 0
    correct_guesses = 0
    perfect_hints = 0
    total_mistakes = 0
    
    game_words = CODENAMES_WORDS
    
    game_batch_size = 20
    
    # play games in batches
    for b in tqdm.tqdm(range(0, number_of_games, game_batch_size)):
        for i in (range(b, b + game_batch_size)):
            start = time.time()
            
            # randomly choose subset size
            subset_size_to_evaluate = random.choice(subset_sizes)
            
            # randomly choose model based on weights
            unnormalized_weights = [weights[-1] for _, weights in models_to_weights.items()]
            normalization_factor = sum(unnormalized_weights)
            weights = [weight / normalization_factor for weight in unnormalized_weights]
            model, type_of_embedding = random.choices(population=models, k=1, weights=weights)[0]
            
            if type_of_embedding == "MULTI-DIM-DEFINITION":
                words_to_embeddings = load_definition_embeddings(model)
            elif type_of_embedding == "MULTI-DIM-CONTEXT":
                words_to_embeddings = load_context_embeddings(model)
                for word in words_to_embeddings:
                    words_to_embeddings[word] = np.vstack(words_to_embeddings[word])
            elif type_of_embedding == "NO MULTI-DIM":
                words_to_embeddings = load_embeddings(model)
                for word in words_to_embeddings:
                    words_to_embeddings[word] = np.array(words_to_embeddings[word]).reshape(1, -1)

            dictionary_words = list(words_to_embeddings.keys())
            game_words = list((set(game_words)).intersection(set(dictionary_words)))
            total_load_time += time.time() - start
            start = time.time()
            board_words, our_words, their_words = initialize_game(game_words)
            (best_hint_subset, best_hint_word) = jack_and_luca_get_best_hint_of_same_size_for_multidefs(words_to_embeddings, dictionary_words, board_words, our_words, their_words, size=subset_size_to_evaluate)
            guessed_words = guess_from_hint(board_words, best_hint_word, subset_size_to_evaluate)
            duration = time.time() - start
            total_time += duration
            
            if len(guessed_words) != subset_size_to_evaluate:
                false_GPT_resp += 1
                continue
            
            intended_guesses_this_round = len(set(best_hint_subset).intersection(set(guessed_words)))
            intended_guesses += intended_guesses_this_round
            correct_guesses += len(set(guessed_words).intersection(set(our_words)))
            total_guesses += subset_size_to_evaluate
            if intended_guesses_this_round == subset_size_to_evaluate:
                perfect_hints += 1
            
            previous_weights = models_to_weights[(model, type_of_embedding)]
            last_weight = previous_weights[-1]
            # current heuristic is to penalize models that give incorrect guesses
            # another heuristic is to penalize unintended guesses of any kind
            incorrect_guesses = len(set(guessed_words).difference(set(our_words)))
            total_mistakes += incorrect_guesses
            weight_penalty_per_mistake = 0.01
            weight_penalty = weight_penalty_per_mistake * incorrect_guesses
            weight_update = 1 - weight_penalty
            new_weight = last_weight * weight_update
            models_to_weights[(model, type_of_embedding)].append(new_weight)
            # pad the other models with the same weight
            for model_name, weights in models_to_weights.items():
                if model_name == (model, type_of_embedding):
                    continue
                their_last_weight = weights[-1]
                weights.append(their_last_weight)
            
            print(f"Model name:", model)
            print(f"Game {i}")
            print(f"Hint: {best_hint_word}")
            print(f"Intended Guesses: {best_hint_subset}")
            print(f"Our words: {our_words}")
            print(f"Their words: {their_words}")
            print(f"Bot guesses: {guessed_words}")
            print(f"Mistakes: {incorrect_guesses}")
            print("\n\n")
            

        old_stdout = sys.stdout
        with open(f"./data/results/experts_framework_2_results.txt", "w") as f:
            sys.stdout = f
            # print weights
            for model, weight in models_to_weights.items():
                print(f"Model: {model}, Weight: {weight}")
            print("\n\n")
            print(f"Average load time: {total_load_time / number_of_games}")
            print(f"Average duration: {total_time / number_of_games}")
            print(f"GPT Misfires: {false_GPT_resp}")
            print(f"Number of games played: {number_of_games}")
            print(f"Correct intended guesses: {intended_guesses}")
            print(f"Correct green guesses: {correct_guesses}")
            print(f"Number of perfect hints given: {perfect_hints}")
            print(f"Total Guesses: {total_guesses}")
            print(f"Intended guesses percentage: {intended_guesses / total_guesses}")
            print(f"Green guesses percentage: {correct_guesses / total_guesses}")
            print(f"Average mistakes: {total_mistakes / number_of_games}")
            print("\n\n\n\n")
        sys.stdout = old_stdout
        
        weights_pickle_file = "./data/results/experts_framework_2_weights.pkl"
        with open(weights_pickle_file, "wb") as f:
            pickle.dump(models_to_weights, f)

if __name__ == "__main__":
    """ Trying out the generalized similarity weighting """
    models_and_types_to_weights = OrderedDict({
        ("bert", "MULTI-DIM-CONTEXT") : .1,
        ("openai", "MULTI-DIM-DEFINITION") : .1,
        # ("word2vec300", "NO MULTI-DIM") : 1,
        # ("glove300", "NO MULTI-DIM") : 1,
        ("fasttext", "NO MULTI-DIM") : 1
    })
    evaluate_generalized_weighted_spymaster_with_guesser_bot(models_and_types_to_weights, 2)
    
    
    """ Trying out the experts framework """
    # # # experts_framework()
    # # # print weights
    # # weights_pickle_file = "./data/results/experts_framework_2_weights.pkl"
    # # with open(weights_pickle_file, "rb") as f:
    # #     models_to_weights = pickle.load(f)
    
    # # model_to_single_weight = { model : weights[-1] for model, weights in models_to_weights.items() }
    # # normalization = sum(model_to_single_weight.values())
    # # model_to_single_weight = { model : weight / normalization for model, weight in model_to_single_weight.items() }
    
    # # for model, weight in model_to_single_weight.items():
    # #     print(f"Model: {model}, Weight: {weight:.2f}")

    """ Testing regular spymaster with guesser bot """
    # type_of_embeddings = ["MULTI-DIM-DEFINITION", "MULTI-DIM-CONTEXT", "NO MULTI-DIM"]

    # type_of_embedding = "NO MULTI-DIM"
    # models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    # model = "fasttext"
    # use_bert_embeddings=True
    # bert_weight=0.1
    
   
    # # type_of_embedding = "MULTI-DIM-CONTEXT"
    # # model_names = ["deberta", "bert", "roberta", "gpt2", "xlnet"]
    # # model = "bert"
    
    # # type_of_embedding = "MULTI-DIM-DEFINITION"
    # # model_names = ["openai"]
    # # model = "openai"

    # subset_size = 2
    # evaluate_spymaster_with_guesser_bot(model, subset_size, use_bert_embeddings=False, bert_weight=bert_weight, type_of_embedding=type_of_embedding)




# Board number: 97
# Hint: nut
# Our words: ['bond', 'code', 'bolt', 'tail', 'antarctica', 'berry', 'thief', 'helicopter']
# Their words: ['alien', 'table', 'cross', 'princess', 'boot', 'board', 'center', 'van']
# Intended Guesses: ['berry', 'bolt']
# Bot guesses: ['berry', 'bolt']
    
# Hint: square
# Our words: ['cat', 'march', 'center', 'lawyer', 'antarctica', 'string', 'circle', 'bottle']
# Their words: ['club', 'date', 'root', 'bat', 'conductor', 'olympus', 'amazon', 'nail']
# Intended Guesses: ['center', 'circle', 'march']






# Duration: 1.1441588401794434
# Board number: 5
# Hint: punch
# Our words: ['box', 'port', 'casino', 'hole', 'round', 'moscow', 'soul', 'laser']
# Their words: ['hand', 'space', 'sock', 'chocolate', 'well', 'horn', 'pistol', 'seal']
# Intended Guesses: ['hole', 'box']
# Bot guesses: ['box']

# Duration: 1.1742370128631592
# Board number: 37
# Hint: wood
# Our words: ['fire', 'forest', 'iron', 'bridge', 'spine', 'egypt', 'plot', 'trunk']
# Their words: ['lemon', 'carrot', 'honey', 'snowman', 'mug', 'pyramid', 'scorpion', 'fly']
# Intended Guesses: ['spine', 'forest']
# Bot guesses: ['forest', 'trunk']

# Duration: 1.246595859527588
# Board number: 30
# Hint: marching
# Our words: ['death', 'root', 'bow', 'press', 'cliff', 'soldier', 'water', 'march']
# Their words: ['hand', 'thief', 'box', 'theater', 'time', 'ray', 'crash', 'fighter']
# Intended Guesses: ['soldier', 'march']
# Bot guesses: ['soldier', 'march']

# Duration: 1.3635430335998535
# Board number: 67
# Hint: wizard
# Our words: ['compound', 'vet', 'contract', 'hood', 'princess', 'nut', 'straw', 'star']
# Their words: ['box', 'torch', 'lion', 'soldier', 'undertaker', 'ruler', 'missile', 'needle']
# Intended Guesses: ['star', 'princess']
# Bot guesses: ['princess', 'star']

# Duration: 0.8866713047027588
# Board number: 166
# Hint: boob
# Our words: ['trip', 'buffalo', 'cell', 'amazon', 'stadium', 'switch', 'shot', 'mouth']
# Their words: ['slip', 'doctor', 'arm', 'rose', 'bed', 'fighter', 'capital', 'net']
# Intended Guesses: ['trip', 'cell']
# Bot guesses: ['rose', 'bed']

# Duration: 1.8424909114837646
# Board number: 207
# Hint: card
# Our words: ['tick', 'sound', 'deck', 'plane', 'washington', 'ketchup', 'switch', 'snowman']
# Their words: ['mercury', 'casino', 'block', 'pan', 'yard', 'check', 'bermuda', 'soldier']
# Intended Guesses: ['snowman', 'deck']
# Bot guesses: ['casino', 'deck']




# Model name: openai
# Game 280
# Hint: pod
# Intended Guesses: ['mail', 'school']
# Our words: ['spy', 'boom', 'microscope', 'mail', 'root', 'school', 'row', 'march']
# Their words: ['gas', 'dwarf', 'belt', 'drill', 'nurse', 'casino', 'dress', 'watch']
# Bot guesses: ['belt', 'drill']
# Mistakes: 2

# Model name: bert
# Game 300
# Hint: batter
# Intended Guesses: ['ball', 'strike', 'pitch']
# Our words: ['rabbit', 'drill', 'turkey', 'strike', 'theater', 'yard', 'ball', 'pitch']
# Their words: ['buck', 'cold', 'ray', 'horseshoe', 'beijing', 'moon', 'shot', 'hole']
# Bot guesses: ['pitch', 'strike', 'ball']
# Mistakes: 0




# fasttext + openAI
# Duration: 0.6857359409332275
# Board number: 14
# Hint: animal
# Intended Guesses: ['shark', 'robot', 'mouse']
# Our words: ['night', 'flute', 'cycle', 'shark', 'mouse', 'opera', 'robot', 'figure']
# Their words: ['hollywood', 'embassy', 'undertaker', 'swing', 'eagle', 'snow', 'america', 'compound']
# Bot guesses: ['mouse', 'eagle', 'shark']

# Duration: 0.6312127113342285
# Board number: 82
# Hint: capricorn
# Intended Guesses: ['atlantis', 'saturn', 'spot']
# Our words: ['atlantis', 'scorpion', 'hollywood', 'spot', 'saturn', 'nut', 'tower', 'plastic']
# Their words: ['ball', 'canada', 'pan', 'washington', 'lab', 'boot', 'jet', 'port']
# Bot guesses: ['saturn', 'scorpion', 'nut']

# Duration: 0.5726170539855957
# Board number: 414
# Hint: star
# Intended Guesses: ['model', 'film', 'hollywood']
# Our words: ['hollywood', 'green', 'unicorn', 'film', 'whip', 'deck', 'embassy', 'model']
# Their words: ['millionaire', 'knife', 'vacuum', 'lead', 'net', 'head', 'chocolate', 'octopus']
# Bot guesses: ['film', 'hollywood', 'model']