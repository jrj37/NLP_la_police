import os
import pandas as pd

# 1. Load CSV data
def load_csv_files(x_ids_path, predilex_path, limit=None):
    x_ids = pd.read_csv(x_ids_path, nrows=limit)
    predilex = pd.read_csv(predilex_path, nrows=limit)
    return x_ids, predilex

# 2. Load text files
def load_text_files(folder_path):
    texts = {}
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts[filename] = file.read()
    return texts

