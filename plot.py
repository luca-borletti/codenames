import matplotlib.pyplot as plt
import numpy as np

def plot():
    models, intended_guesses, green_guesses, perfect_hints = process_data()
    models = ["openai", "word2vec", "glove", "word2vec+glove", "glove100", "glovetwitter", "fasttext"]
    # Set the width of the bars
    bar_width = 0.25

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the bar graph
    plt.bar(r1, intended_guesses, color='b', width=bar_width, edgecolor='grey', label='Correct Intended Guesses')
    plt.bar(r2, green_guesses, color='g', width=bar_width, edgecolor='grey', label='Correct Guesses')
    plt.bar(r3, perfect_hints, color='r', width=bar_width, edgecolor='grey', label='Perfect Hints')

    # Add xticks on the middle of the group bars
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Percentages', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(models))], models, rotation=20)

    # Create legend & Show graphic
    plt.legend()
    plt.title('Performance Over Different Models')
    plt.tight_layout()
    plt.savefig("./data/plots/plots_of_diff_models")
    plt.show()

def process_data():
    models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    intended_guesses_lst, green_guesses_lst, perfect_hints_lst = [], [], []
    for model in models:
        with open(f"./data/results/{model}_results.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Correct intended guesses:" in line:
                    intended_guesses = int(line.split(":")[1].strip())
                elif "Correct green guesses:" in line:
                    green_guesses = int(line.split(":")[1].strip())
                elif "Number of perfect hints given:" in line:
                    perfect_hints = int(line.split(":")[1].strip())
                elif "Total Guesses:" in line:
                    total_guesses = int(line.split(":")[1].strip())
            intended_guesses = intended_guesses / total_guesses
            green_guesses = green_guesses / total_guesses
            perfect_hints = perfect_hints / (total_guesses / 2)
            intended_guesses_lst.append(intended_guesses)
            green_guesses_lst.append(green_guesses)
            perfect_hints_lst.append(perfect_hints)
    return models, intended_guesses_lst, green_guesses_lst, perfect_hints_lst

plot()