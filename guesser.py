"""
Codemaster AI
 
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
import numpy as np
import heapq
from preprocess import load_embeddings, PICKLE_PATH

WORDS_TO_EMBEDDINGS = load_embeddings(PICKLE_PATH)

ALL_WORDS = list(WORDS_TO_EMBEDDINGS.keys())

BOARD_SIZE = 12
OUR_SIZE = 6

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

def cosine_similarity(x, y):
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
    return cosine_similarity(x, y)

def guess_over_all_subsets(words_to_embeddings, all_words, board_words, our_words, their_words):
    """
    Guesses the best word subsets based on their similarity scores.

    Args:
        all_words (list): A list of all possible words.
        board_words (list): A list of words on the game board.
        our_words (list): A list of words that belong to our team.
        their_words (list): A list of words that belong to the opposing team.

    Returns:
        list: A list of the best word subsets, the word to guess, and the 
        worst similarity score between the word to guess and our words.
    """
    best_subsets = []
    n = 5
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
                        if len(best_subsets) > n:
                            heapq.heappop(best_subsets)
            num_subsets += 1
            if num_subsets % 10 == 0:
                print(f"Processed {num_subsets} subsets")
    return best_subsets

def guess_over_each_subset(words_to_embeddings, all_words, board_words, our_words, their_words):
    best_subsets_per_group = []
    n = 5
    num_subsets = 0
    for i in range(2, len(our_words) + 1):
        best_subsets_for_group = []
        heapq.heapify(best_subsets_for_group)
        for subset in subsets(our_words, i):
            for word in all_words:
                if word not in board_words:
                    their_best_sim = max([similarity(words_to_embeddings[word], words_to_embeddings[their_word]) for their_word in their_words])
                    our_worst_sim = min([similarity(words_to_embeddings[word], words_to_embeddings[our_word]) for our_word in subset])
                    if their_best_sim < our_worst_sim:
                        heapq.heappush(best_subsets_for_group, (our_worst_sim, (subset, word)))
                        if len(best_subsets_for_group) > n:
                            heapq.heappop(best_subsets_for_group)
            num_subsets += 1
            if num_subsets % 10 == 0:
                print(f"Processed {num_subsets} subsets")
        best_subsets_per_group.append(best_subsets_for_group)
    return best_subsets_per_group

def start_game(words_to_embeddings, all_words):
    board_words = np.random.choice(all_words, BOARD_SIZE, replace=False)
    our_words = np.random.choice(board_words, OUR_SIZE, replace=False)
    their_words = [word for word in board_words if word not in our_words]
    
    print(f"Board words: {board_words}")
    print(f"Our words: {our_words}")
    print(f"Their words: {their_words}")
    
    return guess_over_each_subset(words_to_embeddings, list(all_words), list(board_words), list(our_words), list(their_words))
    # return guess_over_all_subsets(words_to_embeddings, list(all_words), list(board_words), list(our_words), list(their_words))

if __name__ == "__main__":
    print(start_game(WORDS_TO_EMBEDDINGS, ALL_WORDS))
    # pprint.pprint(WORDS_TO_EMBEDDINGS)
    # pprint.pprint(ALL_WORDS)
    # print(all_subsets(ALL_WORDS))
    # print(len(ALL_WORDS))
    pass

