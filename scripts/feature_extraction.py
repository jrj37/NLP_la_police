import pandas as pd
import re
import spacy
from textblob import TextBlob
from format_funtions import reformat_dates
from transformers import pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Load the French language model
nlp = spacy.load("fr_core_news_sm")

# List of common French first names
male_names = ["Jean", "Pierre", "Paul", "Jacques", "Michel", "Louis", "André", "Henri", "Robert", "Georges", "Philippe"]
female_names = ["Marie", "Jeanne", "Marguerite", "Paulette", "Simone", "Lucie", "Yvonne", "Madeleine", "Hélène", "Marcelle", "Sophie"]

key_words_accident = ["accident", "blessure", "choc", "fracture"]
key_words_consolidation = ["consolidation", "guérison", "rétablissement", "convalescence"]

male_verbs = ["blessé", "accidenté", "décédé", "tombé"]
female_verbs = ["blessée", "accidentée", "décédée", "tombée"]

male_keywords = [
    "père", "garçon", "travailleur", "mari", "monsieur", "époux", 
    "ouvrier", "chauffeur", "artisan", "agriculteur", "cadre", 
    "ingénieur", "apprenti", "jeune homme", "vétéran", "victime masculine"
]
female_keywords = [
    "mère", "fille", "travailleuse", "épouse", "madame", "mademoiselle", 
    "infirmière", "assistante", "employée", "ménagère", "veuve", 
    "jeune femme", "enceinte", "victime féminine"
]

# Mots-clés
accident_keywords = ["accident", "blessure", "choc", "victime", "sinistre", "collision", "incident", "fracture"]
consolidation_keywords = ["consolidation", "guérison", "stabilisation", "rémission", "état final"]


def analyze_sentiment(text):
    """Analyze the sentiment of a text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def count_tokens(tokens, keywords):
    """Count the number of tokens that match a list of keywords."""
    return sum([1 for token in tokens if token in keywords])

def count_gender_markers(text):
    """Count different markers of gender."""
    text_lower = text.lower()
    markers = {
        "il": text_lower.count(" il "),
        "elle": text_lower.count(" elle "),
        "monsieur": text_lower.count("monsieur"),
        "mr": text_lower.count("mr"),
        "madame": text_lower.count("madame"),
        "mme": text_lower.count("mme"),
        "mademoiselle": text_lower.count("mademoiselle"),
        "mlle": text_lower.count("mlle"),
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
        "male_name_count": [],
        "female_name_count": [],
        "male_keywords_count": [],
        "female_keywords_count": [],
        "male_verbs_count": [],
        "female_verbs_count": [],
        "date_accident_pred": [],
        "date_consolidation_pred": [],
        "sentiment_polarity": [],
        "text_length": [],
        "num_words": [],
        "num_sentences": []
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
        
        # Count gendered keywords
        tokens = [token.text for token in nlp(text)]
        # use token for each word in the text
        features["male_name_count"].append(count_tokens(tokens, male_names))
        features["female_name_count"].append(count_tokens(tokens, female_names))
        features["male_keywords_count"].append(count_tokens(tokens, male_keywords))
        features["female_keywords_count"].append(count_tokens(tokens, female_keywords))
        features["male_verbs_count"].append(count_tokens(tokens, male_verbs))
        features["female_verbs_count"].append(count_tokens(tokens, female_verbs))
        
        # Dates
        accident_date, consolidation_date = extract_dates_advanced(text, context_window=10)
        features["date_accident_pred"].append(accident_date)
        features["date_consolidation_pred"].append(consolidation_date)
        
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
    Extract accident and consolidation dates from text with dynamic context.

    Parameters:
    - text (str): Text data.
    - context_window (int): Initial number of words for context extraction.

    Returns:
    - tuple: Extracted accident date and consolidation date.
    """
    date_patterns = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [a-zéû]+ \d{4}|\d{4}-\d{2}-\d{2})\b"
    matches = re.finditer(date_patterns, text)
    
    accident_date, consolidation_date = "n.c.", "n.c."
    words = text.split()


    for match in matches:
        date = match.group()
        start_idx, end_idx = match.start(), match.end()
        
        start_word_idx = max(0, len(text[:start_idx].split()) - context_window)
        end_word_idx = min(len(words), len(text[:end_idx].split()) + context_window)
        context = " ".join(words[start_word_idx:end_word_idx])

        # Classifier selon le contexte
        if any(keyword in context for keyword in accident_keywords):
            accident_date = reformat_dates(date)
        elif any(keyword in context for keyword in consolidation_keywords):
            consolidation_date = reformat_dates(date)
    
    return accident_date, consolidation_date


def extract_dates_with_dependencies(text):
    accident_date, consolidation_date = "n.c.", "n.c."
    doc = nlp(text)
    
    for token in doc:
        # Vérifier si le mot est un mot-clé lié à un accident
        if token.text.lower() in accident_keywords:
            for child in token.children:
                if child.ent_type_ == "DATE":
                    accident_date = child.text
        # Vérifier si le mot est un mot-clé lié à la consolidation
        elif token.text.lower() in consolidation_keywords:
            for child in token.children:
                if child.ent_type_ == "DATE":
                    consolidation_date = child.text
    
    return accident_date, consolidation_date


def extract_dates_with_contextual_expansion(text, context_window=5):
    words = text.split()
    date_patterns = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [a-zéû]+ \d{4}|\d{4}-\d{2}-\d{2})\b"
    matches = re.finditer(date_patterns, text)
    
    accident_date, consolidation_date = "n.c.", "n.c."

    for match in matches:
        date = match.group()
        for expansion in [context_window, context_window * 2]:
            start_idx = max(0, len(text[:match.start()].split()) - expansion)
            end_idx = min(len(words), len(text[:match.end()].split()) + expansion)
            context = " ".join(words[start_idx:end_idx])

            if any(kw in context for kw in accident_keywords):
                accident_date = date
                break
            elif any(kw in context for kw in consolidation_keywords):
                consolidation_date = date
                break

    return accident_date, consolidation_date


def rank_dates_by_relevance(text):
    words = text.split()
    date_patterns = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [a-zéû]+ \d{4}|\d{4}-\d{2}-\d{2})\b"
    matches = re.finditer(date_patterns, text)
    
    scored_dates = []

    for match in matches:
        date = match.group()
        start_idx = max(0, len(text[:match.start()].split()) - 10)
        end_idx = min(len(words), len(text[:match.end()].split()) + 10)
        context = " ".join(words[start_idx:end_idx])
        
        # Calculer un score basé sur la fréquence des mots-clés
        score_accident = sum(context.lower().count(kw) for kw in accident_keywords)
        score_consolidation = sum(context.lower().count(kw) for kw in consolidation_keywords)
        
        scored_dates.append({
            "date": date,
            "score_accident": score_accident,
            "score_consolidation": score_consolidation,
        })

    # Attribuer les dates avec les scores les plus élevés
    accident_date = max(scored_dates, key=lambda x: x["score_accident"])["date"] if scored_dates else "n.c."
    consolidation_date = max(scored_dates, key=lambda x: x["score_consolidation"])["date"] if scored_dates else "n.c."
    
    return accident_date, consolidation_date



def extract_dates_advanced(text, context_window=10):
    # accident_date, consolidation_date = extract_dates_with_contextual_expansion(text, context_window)
    accident_date_ranked, consolidation_date_ranked = rank_dates_by_relevance(text)
    
    
    return reformat_dates(accident_date_ranked), reformat_dates(consolidation_date_ranked)
