"""
Example 7: Leaf Encoding (Tree Embedding) — RandomForest jako preprocessor dla sieci neuronowej.

Technika z papieru:
  "Practical Lessons from Predicting Clicks on Ads at Facebook" (He et al., 2014)

Dataset: mlg-ulb/creditcardfraud (Kaggle)
  284 807 transakcji, ~0.17% to oszustwa (binary imbalanced)

Co demonstruje ten przykład:
  - LeafEncodingStep: RF trenowany na danych treningowych
  - RF.apply() → ID liścia w każdym drzewie → one-hot encoding
  - Połączenie embeddingu z oryginalnymi cechami o wysokiej ważności
  - Sieć neuronowa trenowana na wzbogaconej macierzy cech
  - Porównanie: baseline (scaled) vs leaf_embed workflow

Run:
    python examples/07_leaf_encoding_fraud.py
"""

import numpy as np
import pandas as pd
import kagglehub

from softnet import SoftClassifier
from softnet.workflows import (
    SoftWorkflow,
    LeafEncodingStep,
    list_workflows,
)

# ── 1. Pobierz dane z Kaggle ───────────────────────────────────────────────────
print("Pobieranie danych...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

import os
csv_path = os.path.join(path, "creditcard.csv")
df = pd.read_csv(csv_path)

print(f"Wczytano: {df.shape[0]:,} wierszy, {df.shape[1]} kolumn")
print(f"Oszustwa: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)\n")

X = df.drop(columns=["Class"]).values
y = df["Class"].values

# ── 2. Podejrzyj dostępne workflow presety ─────────────────────────────────────
list_workflows()
print()

# ── 3. Baseline: scaled → soft-net ────────────────────────────────────────────
print("=" * 60)
print("BASELINE: split → scale → fit → validate")
print("=" * 60)

wf_baseline = SoftWorkflow.from_preset("scaled")
result_baseline = wf_baseline.run(
    X, y,
    estimator=SoftClassifier.from_preset("medium", epochs=30, verbose=0),
)

print(f"\nBaseline score (accuracy): {result_baseline.score:.4f}")
print(result_baseline.report)

# ── 4. Leaf Encoding: RF embedding → soft-net ─────────────────────────────────
print("=" * 60)
print("LEAF EMBEDDING: split → scale → leaf_encoding → fit → validate")
print("=" * 60)

# Domyślnie: 100 drzew, 32 liście/drzewo, + wszystkie oryginalne cechy
wf_leaf = SoftWorkflow.from_preset("leaf_embed")
result_leaf = wf_leaf.run(
    X, y,
    estimator=SoftClassifier.from_preset("medium", epochs=30, verbose=0),
)

print(f"\nLeaf Embed score (accuracy): {result_leaf.score:.4f}")
print(result_leaf.report)

# ── 5. Szczegóły embeddingu ────────────────────────────────────────────────────
print("=" * 60)
print("SZCZEGÓŁY EMBEDDINGU")
print("=" * 60)
print(f"Oryginalne cechy:          {result_leaf.ctx['X'].shape[1]}")
print(f"Cechy z liści (one-hot):   {result_leaf.ctx['n_leaf_features']}")
print(f"Łącznie po embeddingu:     {result_leaf.ctx['n_total_features']}")

importances = result_leaf.ctx["leaf_feature_importances"]
top5_idx = np.argsort(importances)[::-1][:5]
print(f"\nTop-5 najważniejszych cech (wg RF):")
feature_names = df.drop(columns=["Class"]).columns.tolist()
for rank, idx in enumerate(top5_idx, 1):
    print(f"  {rank}. {feature_names[idx]:10s}  importance={importances[idx]:.4f}")

# ── 6. Wariant: tylko top-10 cech + embeddingi ────────────────────────────────
print("\n" + "=" * 60)
print("WARIANT: top-10 oryginalnych cech + embeddingi liści")
print("=" * 60)

wf_top10 = SoftWorkflow.from_preset(
    "leaf_embed",
    step_overrides={
        "leaf_encoding": LeafEncodingStep(
            max_original_features=10,
            max_leaf_nodes=32,
            n_estimators=100,
        )
    },
)
result_top10 = wf_top10.run(
    X, y,
    estimator=SoftClassifier.from_preset("medium", epochs=30, verbose=0),
)

print(f"\nTop-10 + leaf score (accuracy): {result_top10.score:.4f}")
print(f"Łącznie cech: {result_top10.ctx['n_total_features']}")

# ── 7. Podsumowanie porównawcze ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PORÓWNANIE WYNIKÓW")
print("=" * 60)
print(f"  Baseline (scaled):        {result_baseline.score:.4f}")
print(f"  Leaf embed (all feats):   {result_leaf.score:.4f}")
print(f"  Leaf embed (top-10):      {result_top10.score:.4f}")
