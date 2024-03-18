import pandas as pd

def process_guesses():
    file_path = "./csv_files/guesses.csv"

    df = pd.read_csv(file_path)
    def process_string(row):
        s = row["base_text"]
        segments = s.split(",")
        guesses = s.split("[")[1].split("]")[0].replace("'", "").replace(" ", "").split(",")
        for segment in segments:
            if 'clue:' in segment:
                clue = segment.split(":")[1].strip()
        return guesses, clue
    
    df["guesses"] = df.apply(lambda x: process_string(x)[0], axis=1)
    df["clue"] = df.apply(lambda x: process_string(x)[1], axis=1)

    def filter_function(row):
        i = row.name
        if i == 0:
            return True
        other_row = df.iloc[i - 1]
        if row["guesses"] == other_row["guesses"] and row["clue"] == other_row["clue"] and len(row["guesses"]) > 1:
            return False
        return True
    
    df = df[df.apply(lambda row: filter_function(row), axis=1)]
    guesses = df["guesses"].tolist()
    clues = df["clue"].tolist()
    return guesses, clues

def process_board():
    file_path = "./csv_files/boards.csv"
    df = pd.read_csv(file_path)
    boards = df["base_text"].tolist()
    return boards

def filter_rows():
    guesses_lst, clues = process_guesses()
    boards = process_board()

    boards_pointer = 0
    remove_indices = []
    i = 0
    while i < len(guesses_lst):
        guesses = guesses_lst[i]
        for guess in guesses:
            if guess not in boards[boards_pointer]:
                remove_indices.append(boards_pointer)
                i -= 1
                break
        i += 1
        boards_pointer += 1
    
    remove_indices = set(remove_indices)
    boards = [boards[i] for i in range(len(boards)) if i not in remove_indices]
    df = pd.DataFrame({
        'guesses': guesses_lst,
        'clues': clues,
        'boards': boards
    })

    def filter_function(row):
        return len(row["guesses"]) > 1
    
    df = df[df.apply(lambda row: filter_function(row), axis=1)]
    return df
            
df = filter_rows()
df.to_csv("./csv_files/GPT_data.csv", index=False)