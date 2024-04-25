""" 
Guesser bot for evaluating spymaster hints and generating guesses.

Uses OpenAI's chat completion API to generate guesses.
"""

import time
import anthropic
from openai import OpenAI
import dotenv
import os
import csv
import re
import ast
import numpy as np
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

DEBUG = True
GAMES_FILE_PATH = "./data/games/hints_and_guesses.csv"

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)

anthropic_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=ANTHROPIC_API_KEY,
)

# message = openai_client.messages.create(
#     model="claude-3-opus-20240229",
#     max_tokens=1000,
#     temperature=0,
#     messages=[]
# )
# print(message.content)


def gpt_4_turbo_guess_from_hint(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=100,
    )
    response_text = response.choices[0].message.content.lower()
    # print(response_text)
    return response_text

def gpt_4_guess_from_hint(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=100,
    )
    response_text = response.choices[0].message.content.lower()
    return response_text

def claude_3_guess_from_hint(prompt):
    response = anthropic_client.completions.create(
        model="claude-3-opus-20240229",
        max_tokens_to_sample=1024,
        prompt=prompt,
    )
    # print(response)
    response_text = response.completion
    return response_text

def gpt_3turbo_guess_from_hint(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=100,
    )
    response_text = response.choices[0].message.content.lower()
    # print(response_text)
    return response_text

def gpt_3_guess_from_hint(prompt):
    response = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0,
    )
    response_text = response.choices[0].text.strip().lower()
    return response_text

def guess_from_hint(board_words, hint, x):
    prompt = f"I am playing codenames. Given the list of words [{', '.join(board_words)}], " \
             f"list {x} words from the list of words that are most related to the hint '{hint}'. " \
             f"You cannot pick {hint}.\n"
    # prompt = f"Given the words {', '.join(board_words)} on the Codenames board and the hint '{hint}', " \
    #          f"list {x} words from the board that are most likely to be related to the hint:\n"

    response_text = gpt_3turbo_guess_from_hint(prompt)
    # response_text = gpt_4_turbo_guess_from_hint(prompt)
    # response_text = gpt_3_guess_from_hint(prompt)
    # response_text = gpt_4_guess_from_hint(prompt)
    filtered_guesses = [word for word in board_words if word in response_text][:x]
    return filtered_guesses

def evaluate_guesser_bot():
    """ 
    Pull human games from the database and evaluate the performance of the guesser bot.
    
    Returns for each subset size group:
        bot_human_fit
        human_performance
        bot_performance
        
    Also return the number of incongruent guesses (when the number of guessed words is not equal to the number of words in the hint)
        
    For each row consisting of guess(es), a clue, and the board state (array for green, black, and tan words)
        Parse the row
        Concatenate all words and randomize the order
        Calculate clue number from # of guesses
        Generate guesses from the hint
        Compare guesses to the board state
        Calculate human-score and green-score
    
    Only evaluate hint groups of 2 and 3
    """
    
    groups = [2, 3]
    tracking = {}
    for group in groups:
        tracking[group] = {
            "total_guesses": 0,
            "bot_human_guesses": 0,
            "bot_green_guesses": 0,
            "human_green_guesses": 0,
            "incongruent_guesses": 0,
        }
    
    num_rows = 0
    total_duration = 0
    
    with open(GAMES_FILE_PATH, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            start = time.time()
            if reader.line_num == 1:
                continue
                
            guess_str = row[0]
            hint_str = row[1]
            board_state_str = row[2]

            str_dict = "{" + re.sub(r'(\w+):', r'"\1":', board_state_str) + "}"
            
            guesses = ast.literal_eval(guess_str)
            hint = hint_str
            board_state = ast.literal_eval(str_dict)
            
            green_words = board_state["green"]
            other_words = board_state["black"] + board_state["tan"]
            board_words = green_words + other_words
            board_words_randomized = np.random.choice(board_words, len(board_words), replace=False)
            board_words = list(board_words_randomized)
            hint_number = len(guesses)
            
            if hint_number not in groups:
                continue
            
            bot_guesses = guess_from_hint(board_words, hint, hint_number)
            
            is_incongruent = 1 if len(bot_guesses) != hint_number else 0
            
            tracking[hint_number]["total_guesses"] += hint_number
            tracking[hint_number]["bot_human_guesses"] += len([x for x in bot_guesses if x in guesses])
            tracking[hint_number]["bot_green_guesses"] += len([x for x in bot_guesses if x in green_words])
            tracking[hint_number]["human_green_guesses"] += len([x for x in guesses if x in green_words])
            tracking[hint_number]["incongruent_guesses"] += is_incongruent

            # if is_incongruent:
                # print(f"INCONGRUENT GUESS")
                # print(f"Hint: {hint}")
                # print(f"Board state: {board_state}")
                # print(f"Guesses: {guesses}")
                # print(f"Bot guesses: {bot_guesses}")
                # print("\n\n")

            if DEBUG:
                print(f"Hint: {hint}")
                print(f"Board state: {board_state}")
                print(f"Guesses: {guesses}")
                print(f"Bot guesses: {bot_guesses}")
                print("\n\n")

                duration = time.time() - start
                total_duration += duration
                
            num_rows += 1
            
            if num_rows % 100 == 0:
                print(f"Processed {num_rows} rows")

    if DEBUG:
        print(f"Average duration: {total_duration / num_rows}")

    results = {}
    for group in groups:
        total_guesses = tracking[group]["total_guesses"]
        bot_human_guesses = tracking[group]["bot_human_guesses"]
        bot_green_guesses = tracking[group]["bot_green_guesses"]
        human_green_guesses = tracking[group]["human_green_guesses"]
        incongruent_guesses = tracking[group]["incongruent_guesses"]
        
        bot_human_fit = bot_human_guesses / total_guesses
        human_performance = human_green_guesses / total_guesses
        bot_performance = bot_green_guesses / total_guesses
        
        results[group] = {
            "bot_human_fit": bot_human_fit,
            "human_performance": human_performance,
            "bot_performance": bot_performance,
            "incongruent_guesses": incongruent_guesses,
        }
    
    return results

if __name__ == "__main__":
    # board_words = ["apple", "car", "chair", "banana", "orange", "person", "dog", "cat", "table", "computer"]
    # hint = "fruit"
    # x = 3
    # print(guess_from_hint(board_words, hint, x))
    
    print(evaluate_guesser_bot())