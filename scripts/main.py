import os
import pandas as pd
from data_preprocessing import load_csv_files, merge_and_preprocess, load_text_files
from feature_extraction import create_feature_dataframe
from evaluation import save_results

LIMIT_FILES = 3

# Define paths
text_folder = "../Data/train_folder_predilex/txt_files/train_folder/txt_files"
x_ids_path = "../Data/train_folder_predilex/txt_files/train_folder/x_train_ids.csv"
predilex_path = "../Data/Y_train_predilex.csv"

def main():
    # Step 1: Load CSV data
    print("Loading CSV data...")
    x_ids, predilex = load_csv_files(x_ids_path, predilex_path, limit=LIMIT_FILES)

    # Step 2: Load text files
    print("Loading text files...")
    texts = load_text_files(text_folder)

    # Step 3: Merge and preprocess data
    print("Merging and preprocessing data...")
    data = merge_and_preprocess(x_ids, predilex)
    
    # # Step 4: Extract features
    print("Extracting features...")
    df = create_feature_dataframe(data,texts)
    
    # # Step 5: Save feature data for modeling
    print("Saving feature data...")
    save_results(df, "../results/df.csv")

if __name__ == "__main__":
    main()
