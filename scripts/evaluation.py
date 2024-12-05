import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(data):
    precision_gender = precision_score(data["real_gender"], data["gender_extracted"], average="weighted", zero_division=1)
    recall_gender = recall_score(data["real_gender"], data["gender_extracted"], average="weighted", zero_division=1)
    f1_gender = f1_score(data["real_gender"], data["gender_extracted"], average="weighted", zero_division=1)
    
    precision_accident_date = precision_score(data["real_accident_date"].notnull(), data["accident_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    recall_accident_date = recall_score(data["real_accident_date"].notnull(), data["accident_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    f1_accident_date = f1_score(data["real_accident_date"].notnull(), data["accident_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    
    precision_consolidation_date = precision_score(data["real_consolidation_date"].notnull(), data["consolidation_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    recall_consolidation_date = recall_score(data["real_consolidation_date"].notnull(), data["consolidation_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    f1_consolidation_date = f1_score(data["real_consolidation_date"].notnull(), data["consolidation_date_extracted"].notnull(), average="binary", pos_label=True, zero_division=1)
    
    metrics = {
        "gender_precision": precision_gender,
        "gender_recall": recall_gender,
        "gender_f1_score": f1_gender,
        "accident_date_precision": precision_accident_date,
        "accident_date_recall": recall_accident_date,
        "accident_date_f1_score": f1_accident_date,
        "consolidation_date_precision": precision_consolidation_date,
        "consolidation_date_recall": recall_consolidation_date,
        "consolidation_date_f1_score": f1_consolidation_date,
    }
    
    return metrics

# 2. Fonction pour sauvegarder les résultats dans un fichier CSV
def save_results(data, output_file):
    # Sauvegarder les données traitées dans un fichier CSV
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
