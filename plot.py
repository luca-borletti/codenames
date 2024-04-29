import matplotlib.pyplot as plt
import numpy as np

# def plot():
#     models, intended_guesses, green_guesses, perfect_hints = process_data()
#     # Set the width of the bars
#     bar_width = 0.25

#     # Set the positions of the bars on the x-axis
#     r1 = np.arange(len(models))
#     r2 = [x + bar_width for x in r1]
#     r3 = [x + bar_width for x in r2]

#     # Create the bar graph
#     plt.bar(r1, intended_guesses, color='b', width=bar_width, edgecolor='grey', label='Correct Intended Guesses')
#     plt.bar(r2, green_guesses, color='g', width=bar_width, edgecolor='grey', label='Correct Guesses')
#     plt.bar(r3, perfect_hints, color='r', width=bar_width, edgecolor='grey', label='Perfect Hints')

#     # Add xticks on the middle of the group bars
#     plt.xlabel('Models', fontweight='bold')
#     plt.ylabel('Percentages', fontweight='bold')
#     plt.xticks([r + bar_width for r in range(len(models))], models, rotation=20)

#     # Create legend & Show graphic
#     plt.legend()
#     plt.title('Performance Over Different Models')
#     plt.tight_layout()
#     plt.savefig("./data/plots/plots_of_diff_models")
#     plt.show()


def plot():
    models, intended_guesses, green_guesses, perfect_hints = process_data()
    # Set the width of the bars
    bar_width = 0.25

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the bar graph
    bars1 = plt.bar(r1, intended_guesses, color='b', width=bar_width, edgecolor='grey', label='Correct Intended Guesses')
    bars2 = plt.bar(r2, green_guesses, color='g', width=bar_width, edgecolor='grey', label='Correct Guesses')
    bars3 = plt.bar(r3, perfect_hints, color='r', width=bar_width, edgecolor='grey', label='Perfect Hints')

    # Function to add labels above the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{round(height * 100, 1)}%', ha='center', va='bottom')

    # Adding data labels
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Add xticks on the middle of the group bars
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Percentages', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(models))], models, rotation=20)

    # Create legend & Show graphic
    plt.legend(loc='lower right', framealpha=0.5)
    plt.title('Performance Over Different Models')
    plt.tight_layout()
    plt.savefig("./data/plots/plots_of_diff_models")
    plt.show()

def process_data():
    models = ["openai", "word2vec300", "glove300", "word2vec+glove300", "glove100", "glovetwitter200", "fasttext"]
    intended_guesses_lst, green_guesses_lst, perfect_hints_lst = [], [], []
    for model in models:
        if model == "fasttext":
            path = "data/results/fasttext_2_results.txt"
        else:
            path = f"data/results/{model}_results.txt"
        with open(path, "r") as f:
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