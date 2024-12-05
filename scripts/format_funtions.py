import spacy
import re
from datetime import datetime

nlp = spacy.load("fr_core_news_sm")


def reformat_dates(date):
    if not date:
        return None
    
    # Dictionary of months in French to numbers
    months = {
        "janvier": "01", "février": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "août": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
    }
    
    # Replace month names with numbers
    for month_name, month_number in months.items():
        date = date.replace(month_name, month_number)

    # Replace date separators to standardize
    date = date.replace("/", "-").replace(" ", "-")
    parts = date.split("-")
    
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"  # Format Y-M-D
    
    return None
