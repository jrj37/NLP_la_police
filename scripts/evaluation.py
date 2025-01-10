import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(data):
    # Harmoniser les valeurs entre sexe et pronoun_gender
    sexe_harmonized = data["sexe"].replace({"homme": "male", "femme": "female"})
    pronoun_gender_harmonized = data["pronoun_gender"]

    # Comparaison directe harmonisée
    correct_gender = sexe_harmonized == pronoun_gender_harmonized
    precision_gender = precision_score(data["sexe"].notnull(), correct_gender, average="binary", pos_label=True, zero_division=1)
    recall_gender = recall_score(data["sexe"].notnull(), correct_gender, average="binary", pos_label=True, zero_division=1)
    f1_gender = f1_score(data["sexe"].notnull(), correct_gender, average="binary", pos_label=True, zero_division=1)

    # Accident date evaluation (comparer les valeurs exactes)
    correct_date_accident = data["date_accident"] == data["date_accident_pred"]
    precision_date_accident = precision_score(data["date_accident"].notnull(), correct_date_accident, average="binary", pos_label=True, zero_division=1)
    recall_date_accident = recall_score(data["date_accident"].notnull(), correct_date_accident, average="binary", pos_label=True, zero_division=1)
    f1_date_accident = f1_score(data["date_accident"].notnull(), correct_date_accident, average="binary", pos_label=True, zero_division=1)

    # Consolidation date evaluation (comparer les valeurs exactes)
    correct_date_consolidation = data["date_consolidation"] == data["date_consolidation_pred"]
    precision_date_consolidation = precision_score(data["date_consolidation"].notnull(), correct_date_consolidation, average="binary", pos_label=True, zero_division=1)
    recall_date_consolidation = recall_score(data["date_consolidation"].notnull(), correct_date_consolidation, average="binary", pos_label=True, zero_division=1)
    f1_date_consolidation = f1_score(data["date_consolidation"].notnull(), correct_date_consolidation, average="binary", pos_label=True, zero_division=1)

    metrics = {
        "gender_precision": precision_gender,
        "gender_recall": recall_gender,
        "gender_f1_score": f1_gender,
        "date_accident_precision": precision_date_accident,
        "date_accident_recall": recall_date_accident,
        "date_accident_f1_score": f1_date_accident,
        "date_consolidation_precision": precision_date_consolidation,
        "date_consolidation_recall": recall_date_consolidation,
        "date_consolidation_f1_score": f1_date_consolidation,
    }
    return metrics

def print_metrics(metrics):
    # Afficher les métriques
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def save_results(data, output_file):
    # Sauvegarder les données traitées dans un fichier CSV
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")