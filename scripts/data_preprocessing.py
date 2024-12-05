import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 2. Merge and preprocess data
def merge_and_preprocess(x_ids, predilex):
    # Merge x_ids and predilex on ID
    data = pd.merge(x_ids, predilex, left_on="ID", right_index=True)
    # Remove columns ID_x and ID_y
    data.drop(columns=["ID_x", "ID_y"], inplace=True)
    return data

def split_data(data, test_size=0.2):
    # Split data into training and testing sets
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test

def handle_missing_data(data):
    # Fill missing values in the dataset
    data.fillna("", inplace=True)
    return data

def load_text_files(folder_path):
    texts = {}
    for _, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts[filename] = file.read()
    return texts

def load_csv_files(x_ids_path, predilex_path, limit=None):
    x_ids = pd.read_csv(x_ids_path, nrows=limit)
    predilex = pd.read_csv(predilex_path, nrows=limit)
    return x_ids, predilex