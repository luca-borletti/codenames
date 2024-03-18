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

CODENAMES_WORDS_FILE_PATH = "./data/words/codenames_words.txt"

def load_codenames_words():
    with open(CODENAMES_WORDS_FILE_PATH, "r") as f:
        words = f.read().lower().splitlines()
    return words

CODENAMES_WORDS = load_codenames_words()

BOARD_SIZE = 16
OUR_SIZE = 8
# BOARD_SIZE = 25
# OUR_SIZE = 9

DEBUG = False

def all_subsets(s):
    """ return all subsets of s """
    if len(s) == 0:
        return [[]]
    all_subsets_without_first = all_subsets(s[1:])
    all_subsets_with_first = [x + [s[0]] for x in all_subsets_without_first]
    return all_subsets_without_first + all_subsets_with_first

def all_nonempty_subsets(s):
    return [x for x in all_subsets(s) if len(x) > 0]

def subsets(s, n):
    """
    Generate all subsets of size n from a given set s.

    Parameters:
    s (list): The input set.
    n (int): The size of subsets to generate.

    Returns:
    list: A list of all subsets of size n.
    """
    if n == 0:
        return []
    if n == 1:
        return [[x] for x in s]
    if len(s) == n:
        return [s]
    all_subsets_without_first = subsets(s[1:], n)
    all_subsets_with_first = [x + [s[0]] for x in subsets(s[1:], n - 1)]
    return all_subsets_without_first + all_subsets_with_first

def cosine_similarity2(x, y):
    """
    Calculates the cosine similarity between two vectors.

    Parameters:
    x (array-like): The first vector.
    y (array-like): The second vector.

    Returns:
    float: The cosine similarity between x and y.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def similarity(x, y):
    return cosine_similarity2(x, y)

def get_hints_over_all_subsets(words_to_embeddings, all_words, board_words, our_words, their_words, top_n = 5):
    """
    Hints the best word subsets based on their similarity scores.

    Args:
        all_words (list): A list of all possible words.
        board_words (list): A list of words on the game board.
        our_words (list): A list of words that belong to our team.
        their_words (list): A list of words that belong to the opposing team.

    Returns:
        list: A list of the best word subsets, the word to hint, and the 
        worst similarity score between the word to hint and our words.
    """
    best_subsets = []
    num_subsets = 0
    heapq.heapify(best_subsets)
    for i in range(2, len(our_words) + 1):
        for subset in subsets(our_words, i):
            for word in all_words:
                if word not in board_words:
                    their_best_sim = max([similarity(words_to_embeddings[word], words_to_embeddings[their_word]) for their_word in their_words])
                    our_worst_sim = min([similarity(words_to_embeddings[word], words_to_embeddings[our_word]) for our_word in subset])
                    if their_best_sim < our_worst_sim:
                        heapq.heappush(best_subsets, (our_worst_sim, (subset, word)))
                        if len(best_subsets) > top_n:
                            heapq.heappop(best_subsets)
            num_subsets += 1
            if num_subsets % 10 == 0:
                print(f"Processed {num_subsets} subsets")
    return best_subsets

def get_hints_over_each_subset(words_to_embeddings, all_words, board_words, our_words, their_words, top_n = 5, biggest_subset = 2):
    best_subsets_per_group = []
    num_subsets = 0
    for i in range(2, biggest_subset + 1):
        best_subsets_for_group = []
        heapq.heapify(best_subsets_for_group)
        for subset in subsets(our_words, i):
            for word in all_words:
                if word not in board_words:
                    their_best_sim = max([similarity(words_to_embeddings[word], words_to_embeddings[their_word]) for their_word in their_words])
                    our_worst_sim = min([similarity(words_to_embeddings[word], words_to_embeddings[our_word]) for our_word in subset])
                    if their_best_sim < our_worst_sim:
                        heapq.heappush(best_subsets_for_group, (our_worst_sim, (subset, word)))
                        if len(best_subsets_for_group) > top_n:
                            heapq.heappop(best_subsets_for_group)
            num_subsets += 1
            if num_subsets % 10 == 0:
                if DEBUG:
                    print(f"Processed {num_subsets} subsets")
        best_subsets_per_group.append(best_subsets_for_group)
    return best_subsets_per_group

def get_best_hint_of_same_size(words_to_embeddings, all_words, board_words, our_words, their_words, size = 2):
    best_hint = None
    best_sim = 0
    num_subsets = 0
    all_subets = subsets(our_words, size)
    
    for subset in all_subets:
        # random_words = np.random.choice(all_words, 1000, replace=False)
        # for word in random_words:
        for word in all_words:
            if word not in board_words:
                their_best_sim = max([similarity(words_to_embeddings[word], words_to_embeddings[their_word]) for their_word in their_words])
                our_worst_sim = min([similarity(words_to_embeddings[word], words_to_embeddings[our_word]) for our_word in subset])
                if their_best_sim < our_worst_sim:
                    if our_worst_sim > best_sim:
                        best_sim = our_worst_sim
                        best_hint = (subset, word)
        num_subsets += 1
        if num_subsets % 10 == 0:
            # if DEBUG:
            print(f"Processed {num_subsets} subsets")
    return best_hint

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

def start_game(words_to_embeddings, all_words):
    board_words = np.random.choice(all_words, BOARD_SIZE, replace=False)
    our_words = np.random.choice(board_words, OUR_SIZE, replace=False)
    their_words = [word for word in board_words if word not in our_words]
    
    print(f"Board words: {board_words}")
    print(f"Our words: {our_words}")
    print(f"Their words: {their_words}")
    
    return get_hints_over_each_subset(words_to_embeddings, list(all_words), list(board_words), list(our_words), list(their_words))
    # return get_hints_over_all_subsets(words_to_embeddings, list(all_words), list(board_words), list(our_words), list(their_words))

def initialize_game(all_words):
    board_words = np.random.choice(all_words, BOARD_SIZE, replace=False)
    our_words = np.random.choice(board_words, OUR_SIZE, replace=False)
    their_words = [word for word in board_words if word not in our_words]
    return list(board_words), list(our_words), list(their_words)

def evaluate_spymaster_with_guesser_bot(model):
    number_of_games = 500
    # subsets_to_evaluate = [2,3]
    subset_size_to_evaluate = 2
    game_words = CODENAMES_WORDS
    words_to_embeddings = load_embeddings(model)
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
        
        if len(guessed_words) < subset_size_to_evaluate:
            false_GPT_resp += 1
            continue
        
        intended_guesses_this_round = len(set(best_hint_subset).intersection(set(guessed_words)))
        intended_guesses += intended_guesses_this_round
        correct_guesses += len(set(guessed_words).intersection(set(our_words)))
        total_guesses += subset_size_to_evaluate
        if intended_guesses_this_round == subset_size_to_evaluate:
            perfect_hints += 1
        # board_words, our_words, their_words = initialize_game(game_words)
        # best_hint_for_each_subset = get_hints_over_each_subset(words_to_embeddings, dictionary_words, board_words, our_words, their_words, top_n=1, biggest_subset=BIGGEST_SUBSET)
        # best_hint_for_each_subset = [x[0][1] for x in best_hint_for_each_subset]
        # best_hint_for_each_subset = [(x[1], len(x[0]), x[0]) for x in best_hint_for_each_subset]
        # print(f"Game {i + 1}: {best_hint_for_each_subset}")
        # for hint, subset_size, actual_words in best_hint_for_each_subset:
        #     guessed_words = guess_from_hint(board_words, hint, subset_size)
            # results.append((subset_size, actual_words, guessed_words))
            # total_guesses += subset_size
            # correct_guesses += len([x for x in guessed_words if x in actual_words])
            # print(f"Guessed words: {guessed_words} for hint {hint} and actual words {actual_words}")
        # print("\n\n")
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
    # print(load_codenames_words())
    # print(evaluate_with_guesser_bot(ALL_WORDS, WORDS_TO_EMBEDDINGS))
    # print(start_game(WORDS_TO_EMBEDDINGS, ALL_WORDS))
    # pprint.pprint(WORDS_TO_EMBEDDINGS)
    # pprint.pprint(ALL_WORDS)
    # print(all_subsets(ALL_WORDS))
    # print(len(ALL_WORDS))
    model = "glove"
    evaluate_spymaster_with_guesser_bot(model)



""" 
Use codenames 400 to choose from for board

Evalute ChatGPT


"""