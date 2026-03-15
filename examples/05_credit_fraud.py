"""
Example 5: Credit Card Fraud Detection — klasyfikacja binarna na niezbalansowanym zbiorze.

Dataset: mlg-ulb/creditcardfraud (Kaggle)
  284 807 transakcji, tylko ~0.17% to oszustwa (klasa 1)
  Cechy V1–V28: wyniki PCA (anonimowe), plus Time i Amount

Co demonstruje ten przykład:
  - kagglehub do pobrania danych
  - SoftClassifier automatycznie wykrywa binary → sigmoid + binary_crossentropy
  - obsługa niezbalansowanych klas przez class_weight
  - predict_proba + próg decyzyjny zamiast domyślnego 0.5
  - raport klasyfikacji (precision, recall, F1)

Run:
    python examples/05_credit_fraud.py
"""

import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from softnet import SoftClassifier

# ── 1. Pobierz dane z Kaggle ───────────────────────────────────────────
print("Pobieranie danych...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")

print(f"Załadowano {len(df):,} transakcji")
print(f"Oszustwa: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")

# ── 2. Przygotowanie cech ─────────────────────────────────────────────
# V1–V28 są już po PCA, skalujemy tylko Time i Amount
df["Amount_scaled"] = StandardScaler().fit_transform(df[["Amount"]])
df["Time_scaled"]   = StandardScaler().fit_transform(df[["Time"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols].values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ── 3. Wagi klas — kompensacja niezbalansowania ───────────────────────
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
weight_for_fraud = n_neg / n_pos   # ~580x wyższa waga dla klasy 1
class_weight = {0: 1.0, 1: weight_for_fraud}
print(f"Waga klasy 'oszustwo': {weight_for_fraud:.0f}x")

# ── 4. Model — zero konfiguracji ──────────────────────────────────────
# SoftClassifier sam wykrywa: y ∈ {0,1} → binary task
clf = SoftClassifier(
    layers=[64, 32],
    dropout=0.3,
    batch_norm=True,
    epochs=30,
    batch_size=512,    # duży batch — zbiór jest duży
    patience=5,
    verbose=1,
)
clf.fit(X_train, y_train)

print("\n--- Dlaczego te wartości domyślne? ---")
print(clf.explain())

# ── 5. Ewaluacja ──────────────────────────────────────────────────────
# predict_proba zamiast predict — lepszy próg dla niezbalansowanych danych
proba = clf.predict_proba(X_test)[:, 1]   # P(fraud)

# dla fraudu ważniejszy recall niż precision — próg 0.3 zamiast 0.5
threshold = 0.3
y_pred_thresh = (proba >= threshold).astype(int)

print(f"\n=== Wyniki (próg = {threshold}) ===")
print(classification_report(y_test, y_pred_thresh, target_names=["Normalna", "Oszustwo"]))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

cm = confusion_matrix(y_test, y_pred_thresh)
tn, fp, fn, tp = cm.ravel()
print(f"\nMacierz pomyłek:")
print(f"  Wykryte oszustwa (TP):     {tp}")
print(f"  Przeoczone oszustwa (FN):  {fn}")
print(f"  Fałszywy alarm (FP):       {fp}")
print(f"  Poprawne normalne (TN):    {tn:,}")
