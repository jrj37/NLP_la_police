import spacy
import re
from datetime import datetime

# Charger le modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")


# 3. Fonction pour reformater les dates (par exemple de "1 janvier 2020" à "2020-01-01")
def reformat_dates(date):
    if not date:
        return None
    
    # Dictionnaire de mois en français à numéro
    months = {
        "janvier": "01", "février": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "août": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
    }
    
    # Remplacer les mois en texte par leurs numéros
    for month_name, month_number in months.items():
        date = date.replace(month_name, month_number)

    # Remplacer les séparateurs de date pour uniformiser
    date = date.replace("/", "-").replace(" ", "-")
    parts = date.split("-")
    
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"  # Format Y-M-D
    
    return None
