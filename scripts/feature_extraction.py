import pandas as pd
import re
import spacy
from textblob import TextBlob
from format_funtions import reformat_dates

# Charger le modèle SpaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Liste de prénoms courants masculins et féminins
male_names = ["Jean", "Pierre", "Paul", "Jacques", "Michel", "Louis", "André", "Henri", "Robert", "Georges", "Philippe"]
female_names = ["Marie", "Jeanne", "Marguerite", "Paulette", "Simone", "Lucie", "Yvonne", "Madeleine", "Hélène", "Marcelle", "Sophie"]

def analyze_sentiment(text):
    """Analyse de sentiment d'un texte avec TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity


def count_gender_markers(text):
    """Compte les marqueurs masculins et féminins dans un texte."""
    text_lower = text.lower()
    markers = {
        "il": text_lower.count(" il "),
        "elle": text_lower.count(" elle "),
        "monsieur": text_lower.count("monsieur"),
        "madame": text_lower.count("madame"),
        "mr": text_lower.count("mr"),
        "mme": text_lower.count("mme")
    }
    
    # Déterminer le genre dominant pour chaque paire
    gender_dominance = {
        "pronoun_gender": "male" if markers["il"] > markers["elle"] else "female" if markers["elle"] > markers["il"] else "neutral",
        "title_gender": "male" if markers["monsieur"] + markers["mr"] > markers["madame"] + markers["mme"] else "female" if markers["madame"] + markers["mme"] > markers["monsieur"] + markers["mr"] else "neutral"
    }
    
    return markers, gender_dominance

def create_feature_dataframe(data, text_dict):
    """
    Crée un DataFrame contenant des caractéristiques (features) extraites et les variables cibles.
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
        
        # Comptage des marqueurs de genre
        markers, dominance = count_gender_markers(text)
        features["il_count"].append(markers["il"])
        features["elle_count"].append(markers["elle"])
        features["monsieur_count"].append(markers["monsieur"])
        features["madame_count"].append(markers["madame"])
        features["pronoun_gender"].append(dominance["pronoun_gender"])
        features["title_gender"].append(dominance["title_gender"])
        
        # Noms
        tokens = [token.text for token in nlp(text)]
        features["contains_male_name"].append(1 if any(name in tokens for name in male_names) else 0)
        features["contains_female_name"].append(1 if any(name in tokens for name in female_names) else 0)
        
        # Verbes
        verbs = [token.lemma_ for token in nlp(text) if token.pos_ == "VERB"]
        features["contains_verb_tomber"].append(1 if "tomber" in verbs else 0)
        features["contains_verb_consolider"].append(1 if "consolider" in verbs else 0)
        
        # Dates
        accident_date, consolidation_date = extract_dates_with_context(text, context_window=10)
        features["accident_date"].append(accident_date)
        features["consolidation_date"].append(consolidation_date)
        
        # Mots-clés
        keywords = ["accident", "consolidation", "blessure", "choc", "fracture"]
        features["keyword_count"].append(sum(text.lower().count(word) for word in keywords))
        
        # Sentiment
        features["sentiment_polarity"].append(analyze_sentiment(text))
        
        # Métadonnées textuelles
        features["text_length"].append(len(text))
        features["num_words"].append(len(text.split()))
        features["num_sentences"].append(len(re.split(r"[.!?]", text)))
    
    feature_df = pd.DataFrame(features)
    combined_df = pd.concat([data.reset_index(drop=True), feature_df], axis=1)
    
    return combined_df


def extract_dates_with_context(text, context_window=10):
    """
    Extraction de dates avec des indices contextuels dans une fenêtre de texte.
    
    Parameters:
    - text (str): Texte à analyser.
    - context_window (int): Nombre de mots autour de la date à extraire pour le contexte.
    
    Returns:
    -  The most relevant date extracted for accident and consolidation.
    """
    # Modèle de date
    date_patterns = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [a-zéû]+ \d{4}|\d{4}-\d{2}-\d{2})\b"
    matches = re.finditer(date_patterns, text)
    
    accident_date = "n.c."
    consolidation_date = "n.c."

    # Découper le texte en mots
    words = text.split()
    
    for match in matches:
        date = match.group()
        start_idx, end_idx = match.start(), match.end()
        
        # Trouver les indices des mots entourant la date
        start_word_idx = max(0, len(text[:start_idx].split()) - context_window)
        end_word_idx = min(len(words), len(text[:end_idx].split()) + context_window)
        
        # Extraire le contexte autour de la date
        context = " ".join(words[start_word_idx:end_word_idx])
        
        # key word for date extraction one line 
        if "accident" in context:
            accident_date = reformat_dates(date)
        elif "consolidation" in context:
            consolidation_date = reformat_dates(date)
        
    
    return accident_date, consolidation_date
