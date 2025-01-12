import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Charger les données
data = pd.read_csv("results/df.csv")

# Vérifier les données
print(data.head())

# Colonnes utiles pour la prédiction
useful_columns = [
    "il_count", "elle_count", "monsieur_count", "madame_count",
    "male_name_count", "female_name_count","male_keywords_count",
    "female_keywords_count", "male_verbs_count",
    "female_verbs_count", "sentiment_polarity", "text_length",
    "num_words", "num_sentences"
]

# Définir les features (X) et la cible (y)
X = data[useful_columns]
y = data["sexe"]

# Encoder la cible (sexe)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Transforme 'homme', 'femme', 'n.c.' en valeurs numériques

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Construire le pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Imputation des valeurs manquantes
    ("scaler", StandardScaler()),                # Standardisation des données
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))  # Modèle de classification
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Évaluer le modèle sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Décoder les prédictions et les cibles pour afficher les labels originaux
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Afficher le rapport de classification
print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Afficher la matrice de confusion
print("Confusion Matrix:")
print(confusion_matrix(y_test_decoded, y_pred_decoded))

# Afficher la précision globale
print("Accuracy:", accuracy_score(y_test_decoded, y_pred_decoded))

# Sauvegarder le modèle
import joblib
joblib.dump(pipeline, "sexe_prediction_model.pkl")
