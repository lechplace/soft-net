"""
Example 6: Iris — klasyfikacja wieloklasowa (3 gatunki).

Dataset: uciml/iris (Kaggle)
  150 próbek, 4 cechy (wymiary płatków i działek kielicha), 3 klasy:
  Iris-setosa, Iris-versicolor, Iris-virginica

Co demonstruje ten przykład:
  - kagglehub do pobrania danych
  - SoftClassifier automatycznie wykrywa multiclass → softmax + sparse_categorical_crossentropy
  - predict_proba zwraca (n, 3) — prawdopodobieństwo każdej klasy
  - explain() pokazuje uzasadnienie wyboru konfiguracji
  - porównanie z domyślnym modelem sklearn (LogisticRegression)

Run:
    python examples/06_iris_multiclass.py
"""

import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from softnet import SoftClassifier

# ── 1. Pobierz dane z Kaggle ───────────────────────────────────────────
print("Pobieranie danych...")
path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(f"{path}/Iris.csv")

print(f"Załadowano {len(df)} próbek")
print(f"Klasy: {df['Species'].unique()}")

# ── 2. Przygotowanie cech ─────────────────────────────────────────────
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values
y = df["Species"].values   # string labels — SoftClassifier obsługuje je automatycznie

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 3. soft-net — zero konfiguracji ──────────────────────────────────
# SoftClassifier widzi 3 unikalne klasy → multiclass
clf = SoftClassifier(
    layers=[32, 16],
    dropout=0.1,
    epochs=100,
    patience=15,
    verbose=1,
)
clf.fit(X_train, y_train)

print("\n--- Dlaczego te wartości domyślne? ---")
print(clf.explain())

# ── 4. Ewaluacja ──────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print(f"\n=== Wyniki soft-net ===")
print(classification_report(y_test, y_pred))

# predict_proba → (n, 3): rozkład prawdopodobieństwa po klasach
proba = clf.predict_proba(X_test)
print("Przykładowe prawdopodobieństwa (pierwsze 5 próbek):")
species = clf.classes_
header = "  " + "  ".join(f"{s:<20}" for s in species)
print(header)
for i in range(5):
    row = "  " + "  ".join(f"{p:<20.3f}" for p in proba[i])
    print(row)

# ── 5. Porównanie z LogisticRegression ────────────────────────────────
print("\n=== Porównanie z sklearn LogisticRegression ===")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
sn_score = clf.score(X_test, y_test)

print(f"  LogisticRegression accuracy: {lr_score:.4f}")
print(f"  SoftClassifier accuracy:     {sn_score:.4f}")

# ── 6. Cross-validation (sklearn CV działa z SoftClassifier) ─────────
# Uwaga: iris jest mały (150 próbek), CV daje lepszy obraz niż single split
print("\n=== Cross-validation (5-fold) — SoftClassifier ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# tworzymy świeży model do CV (bez wcześniejszego fitu)
clf_cv = SoftClassifier(layers=[32, 16], dropout=0.1, epochs=100, patience=15)
scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="accuracy")
print(f"  Accuracy per fold: {scores.round(3)}")
print(f"  Mean ± std: {scores.mean():.3f} ± {scores.std():.3f}")
