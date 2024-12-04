import os
import pandas as pd
import re
from load_data import load_csv_files, load_text_files
import spacy
from datetime import datetime

nlp = spacy.load("fr_core_news_sm")

male_names = ["Jean", "Pierre", "Paul", "Jacques", "Michel", "Louis", "André", "Henri", "Robert", "Georges"]
female_names = ["Marie", "Jeanne", "Marguerite", "Paulette", "Simone", "Lucie", "Yvonne", "Madeleine", "Hélène", "Marcelle"]

LIMIT_FILES = 3

# Define paths
text_folder = "Data/train_folder_predilex/txt_files/train_folder/txt_files"
x_ids_path = "Data/train_folder_predilex/txt_files/train_folder/x_train_ids.csv"
predilex_path = "Data/Y_train_predilex.csv"

# 1. Load data
def load_data(x_ids_path, predilex_path, limit=None):
    x_ids, predilex = load_csv_files(x_ids_path, predilex_path, limit=limit)
    return x_ids, predilex

# 2. Merge and preprocess data
def merge_and_preprocess(x_ids, predilex):
    # Merge x_ids and predilex on ID
    data = pd.merge(x_ids, predilex, left_on="ID", right_index=True)
    # Remove columns ID_x and ID_y
    data.drop(columns=["ID_x", "ID_y"], inplace=True)
    return data

def find_gender_with_nlp(text):
    # Apply spaCy to process the text
    doc = nlp(text)
    
    # Search for titles "Monsieur" or "Madame"
    if "Monsieur" in text:
        return "homme"
    elif "Madame" in text:
        return "femme"

    # Analyze first names in the text
    for token in doc:
        if token.text in male_names:
            return "homme"
        elif token.text in female_names:
            return "femme"
    
    # If no gender can be determined, return "unknown"
    return "n.c."

def validate_date(date):
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        if 1900 <= parsed_date.year <= datetime.now().year:
            return True
    except ValueError:
        return False
    return False

def reformat_dates(date):
    if not date:
        return None
    
    # (January -> 01, February -> 02, etc.)
    months = {
        "janvier": "01", "février": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "août": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
    }
    
    # Replace month names with numbers
    for month_name, month_number in months.items():
        date = date.replace(month_name, month_number)

    # Final format: Y-M-D
    date = date.replace("/", "-").replace(" ", "-")
    parts = date.split("-")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"  # format Y-M-D
    return None

# 3. Extract gender and dates from text files
def extract_gender_and_dates(text):
    # Default values
    gender = None
    accident_date = None
    consolidation_date = None

    # Function to find gender
    gender = find_gender_with_nlp(text)

    # Find dates with regex: \b(?:\d{1,2} [a-zéû]+ \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b
    dates = re.findall(r"\b(?:\d{1,2} [a-zéû]+ \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b", text)
    
    if len(dates) >= 2:
        accident_date = dates[0]
        consolidation_date = dates[1]
    elif len(dates) == 1:
        accident_date = dates[0]

    accident_date = reformat_dates(accident_date)
    consolidation_date = reformat_dates(consolidation_date)
    
    return gender, accident_date, consolidation_date

# 4. Process and extract information
def process_texts(data, texts):
    results = []
    print(f"Processing {len(data)} rows...")
    for _, row in data.iterrows():
        print(f"Processing file: {row['filename']}")
        
        # Use the filename to fetch the text content from the dictionary
        file_text = texts.get(row["filename"], "")  # Use .get to avoid KeyError
        if not file_text:
            print(f"Warning: Text not found for file {row['filename']}. Skipping.")
            continue
        
        # Extract information
        gender, accident_date, consolidation_date = extract_gender_and_dates(file_text)
        
        # Append results
        results.append({
            "ID": row["ID"],
            "filename": row["filename"],
            "real_gender": row["sexe"],
            "gender_extracted": gender,
            "real_accident_date": row["date_accident"],
            "accident_date_extracted": accident_date,
            "real_consolidation_date": row["date_consolidation"],
            "consolidation_date_extracted": consolidation_date
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Step 1: Load CSV data
    print("Loading CSV data...")
    x_ids, predilex = load_data(x_ids_path, predilex_path, limit=LIMIT_FILES)

    # Step 2: Merge and preprocess data
    print("Merging data...")
    data = merge_and_preprocess(x_ids, predilex)

    # Step 3: Load text files
    print("Loading text files...")
    texts = load_text_files(text_folder)

    # Step 4: Extract gender and dates
    print("Extracting information from texts...")
    processed_data = process_texts(data, texts)

    # Debug: Show extracted data
    print("Extracted Data:")
    print(processed_data.head())

    # Save processed data for further analysis
    output_csv = "results/processed_data.csv"
    processed_data.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")
