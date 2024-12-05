import pandas as pd
import re
import spacy
from textblob import TextBlob
from format_funtions import reformat_dates

# Load the French language model
nlp = spacy.load("fr_core_news_sm")

# List of common French first names
male_names = ["Jean", "Pierre", "Paul", "Jacques", "Michel", "Louis", "André", "Henri", "Robert", "Georges", "Philippe"]
female_names = ["Marie", "Jeanne", "Marguerite", "Paulette", "Simone", "Lucie", "Yvonne", "Madeleine", "Hélène", "Marcelle", "Sophie"]

def analyze_sentiment(text):
    """Analyze the sentiment of a text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity


def count_gender_markers(text):
    """Count different markers of gender."""
    text_lower = text.lower()
    markers = {
        "il": text_lower.count(" il "),
        "elle": text_lower.count(" elle "),
        "monsieur": text_lower.count("monsieur"),
        "mr": text_lower.count("mr"),
        "m." : text_lower.count("m."),
        "madame": text_lower.count("madame"),
        "mme": text_lower.count("mme"),
        "mme.": text_lower.count("mme.")
    }
    
    # Find the dominant gender based on the counts
    gender_dominance = {
        "pronoun_gender": "male" if markers["il"] > markers["elle"] else "female" if markers["elle"] > markers["il"] else "neutral",
        "title_gender": "male" if markers["monsieur"] + markers["mr"] > markers["madame"] + markers["mme"] else "female" if markers["madame"] + markers["mme"] > markers["monsieur"] + markers["mr"] else "neutral"
    }
    
    return markers, gender_dominance

def create_feature_dataframe(data, text_dict):
    """
    Create a DataFrame with extracted features from the text data.

    Parameters:
    - data (pd.DataFrame): DataFrame with metadata.
    - text_dict (dict): Dictionary with text data, where the keys are the filenames.

    Returns:
    - pd.DataFrame: DataFrame with extracted features.
    """
    features = {
        "il_count": [],
        "elle_count": [],
        "monsieur_count": [],
        "madame_count": [],
        "pronoun_gender": [],
        "title_gender": [],
        "contains_male_name": [],
        "contains_female_name": [],
        "contains_verb_tomber": [],
        "contains_verb_consolider": [],
        "accident_date": [],
        "consolidation_date": [],
        "keyword_count": [],
        "sentiment_polarity": [],
        "text_length": [],
        "num_words": [],
        "num_sentences": [],
    }
    
    for filename in data["filename"]:
        text = text_dict.get(filename, "")
        
        # Gender markers and dominance
        markers, dominance = count_gender_markers(text)
        features["il_count"].append(markers["il"])
        features["elle_count"].append(markers["elle"])
        features["monsieur_count"].append(markers["monsieur"])
        features["madame_count"].append(markers["madame"])
        features["pronoun_gender"].append(dominance["pronoun_gender"])
        features["title_gender"].append(dominance["title_gender"])
        
        # Names
        tokens = [token.text for token in nlp(text)]
        features["contains_male_name"].append(1 if any(name in tokens for name in male_names) else 0)
        features["contains_female_name"].append(1 if any(name in tokens for name in female_names) else 0)
        
        # Verbs
        verbs = [token.lemma_ for token in nlp(text) if token.pos_ == "VERB"]
        features["contains_verb_tomber"].append(1 if "tomber" in verbs else 0)
        features["contains_verb_consolider"].append(1 if "consolider" in verbs else 0)
        
        # Dates
        accident_date, consolidation_date = extract_dates_with_context(text, context_window=10)
        features["accident_date"].append(accident_date)
        features["consolidation_date"].append(consolidation_date)
        
        # Keywords
        keywords = ["accident", "consolidation", "blessure", "choc", "fracture"]
        features["keyword_count"].append(sum(text.lower().count(word) for word in keywords))
        
        # Sentiment analysis
        features["sentiment_polarity"].append(analyze_sentiment(text))
        
        # Metadata
        features["text_length"].append(len(text))
        features["num_words"].append(len(text.split()))
        features["num_sentences"].append(len(re.split(r"[.!?]", text)))
    
    feature_df = pd.DataFrame(features)
    combined_df = pd.concat([data.reset_index(drop=True), feature_df], axis=1)
    
    return combined_df


def extract_dates_with_context(text, context_window=10):
    """
    Extract accident and consolidation dates from text with context.

    Parameters:
    - text (str): Text data.
    - context_window (int): Number of words to include before and after the date.

    Returns:
    - tuple: Extracted accident date and consolidation date.
    """
    # Modified date patterns to include more date formats
    date_patterns = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [a-zéû]+ \d{4}|\d{4}-\d{2}-\d{2})\b"
    matches = re.finditer(date_patterns, text)
    
    accident_date = "n.c."
    consolidation_date = "n.c."

    # Split the text into words for context extraction
    words = text.split()
    
    for match in matches:
        date = match.group()
        start_idx, end_idx = match.start(), match.end()
        
        # Define the start and end word indices for the context window
        start_word_idx = max(0, len(text[:start_idx].split()) - context_window)
        end_word_idx = min(len(words), len(text[:end_idx].split()) + context_window)
        
        # Extract the context window
        context = " ".join(words[start_word_idx:end_word_idx])
        
        # Check the context for relevant keywords
        if "accident" in context:
            accident_date = reformat_dates(date)
        elif "consolidation" in context:
            consolidation_date = reformat_dates(date)
        
    
    return accident_date, consolidation_date
