"""
Przykład 08 — SoftPipeline: train → save → load → predict (produkcja)
======================================================================

Demonstruje cały cykl produkcyjny:

  DEV:  wf.run(X, y)  → result.to_pipeline() → pipe.save()
  PROD: SoftPipeline.load() → pipe.predict(X_new)   ← brak y, brak trenowania

Dataset: UCI Credit Card Fraud (Kaggle creditcard.csv)
         284 807 transakcji, 492 fraud (0.17%), 30 cech PCA (V1–V28, Amount, Time)

Pobieranie danych (jednorazowo):
    kaggle datasets download -d mlg-ulb/creditcardfraud
    unzip creditcardfraud.zip -d /tmp/
"""

# ── 0. import i konfiguracja ──────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from softnet import SoftClassifier, SoftPipeline
from softnet.workflows import SoftWorkflow

DATASET_PATH = Path("/tmp/creditcard.csv")
MODEL_DIR    = Path("/tmp/soft-net-fraud-v1")        # katalog pipeline
MODEL_ZIP    = Path("/tmp/soft-net-fraud-v1.softpipe")  # .softpipe (zip)

print("=" * 60)
print("  soft-net — Pipeline save / load demo")
print("=" * 60)

# ── 1. Dane ───────────────────────────────────────────────────────────────────

if DATASET_PATH.exists():
    print(f"\n[1/5] Wczytywanie danych z {DATASET_PATH} …")
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.int32)
    print(f"      Shape: {X.shape}  |  Fraud: {y.sum()} ({y.mean()*100:.2f}%)")
else:
    # Syntetyczne dane — do testów bez Kaggle
    print("\n[1/5] Brak creditcard.csv — generuję syntetyczne dane (n=5000) …")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=5_000, n_features=30, n_informative=15,
        n_classes=2, weights=[0.98, 0.02], random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    print(f"      Shape: {X.shape}  |  Fraud: {y.sum()} ({y.mean()*100:.2f}%)")

# ── 2. Trenowanie na DEV ──────────────────────────────────────────────────────

print("\n[2/5] Trenowanie workflow (scale → leaf_encoding → fit) …")

clf = SoftClassifier.from_preset(
    "imb_funnel_heavy",   # [512, 256, 128, 64], dropout=0.4, BN=True
    epochs=30,
    verbose=0,
)

wf = SoftWorkflow.from_preset("imb_leaf_robust")   # split + RobustScaler + leaf + fit + validate

result = wf.run(X, y, estimator=clf)

print(f"\n      Accuracy : {result.score:.4f}")
if result.ctx.get("validation", {}).get("roc_auc"):
    print(f"      AUC-ROC  : {result.ctx['validation']['roc_auc']:.4f}")
print(f"\n{result.report}")

# ── 3. Konwersja i zapis pipeline ─────────────────────────────────────────────

print("[3/5] Eksport do SoftPipeline …")
pipe = result.to_pipeline()
pipe.summary()

# Zapis jako katalog (do lokalnego wdrożenia / debug)
pipe.save(MODEL_DIR)

# Zapis jako .softpipe (zip) — jeden plik do Cloud Run / Docker
pipe.save(MODEL_DIR, as_zip=True)

print(f"\n      Katalog : {MODEL_DIR}/")
print(f"      Zip     : {MODEL_ZIP}")

# ── 4. Załadowanie pipeline (symulacja PROD) ──────────────────────────────────

print("\n[4/5] Ładowanie pipeline z dysku (symulacja prod) …")

# Z katalogu
pipe_loaded = SoftPipeline.load(MODEL_DIR)
print(f"      Załadowano (katalog): {pipe_loaded}")

# Z .softpipe
pipe_zip = SoftPipeline.load(MODEL_ZIP)
print(f"      Załadowano (zip)    : {pipe_zip}")

pipe_zip.summary()

# ── 5. Inferencja na PROD — brak y, brak trenowania ──────────────────────────

print("\n[5/5] Inferencja na nowych danych (brak y) …")

# Symuluj "nowe dane z serwera produkcyjnego"
# W praktyce: X_new = pd.read_csv("new_transactions.csv").values
rng   = np.random.default_rng(0)
idx   = rng.choice(len(X), size=200, replace=False)
X_new = X[idx]                   # surowe dane (przed skalowaniem) — identyczne jak na treningu

# predict — klasy (0/1)
y_pred = pipe_zip.predict(X_new)
print(f"\n      predict()  → klasy: {np.bincount(y_pred.astype(int))} (0=ok, 1=fraud)")

# predict_proba — prawdopodobieństwo fraudu
proba = pipe_zip.predict_proba(X_new)
fraud_score = proba[:, 1]
print(f"      predict_proba()[:, 1]  → min={fraud_score.min():.3f}, "
      f"max={fraud_score.max():.3f}, mean={fraud_score.mean():.3f}")

# decision_score — surowy output sieci (do własnego progowania)
raw = pipe_zip.decision_score(X_new)
print(f"      decision_score() shape : {raw.shape}")

# Własny próg decyzji (ważne dla imbalanced!)
custom_threshold = 0.3   # niższy próg = więcej wykrytych fraudów, więcej false positive
y_custom = (fraud_score >= custom_threshold).astype(int)
print(f"\n      Próg 0.5 → fraudów: {y_pred.sum()}")
print(f"      Próg 0.3 → fraudów: {y_custom.sum()}")

# ── Podsumowanie ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Podsumowanie workflow produkcyjnego")
print("=" * 60)
print("""
DEV  (trening):
  wf     = SoftWorkflow.from_preset("imb_leaf_robust")
  clf    = SoftClassifier.from_preset("imb_funnel_heavy", epochs=30)
  result = wf.run(X, y, estimator=clf)

  pipe   = result.to_pipeline()
  pipe.save("models/fraud_v1", as_zip=True)   → fraud_v1.softpipe

PROD (inferencja):
  from softnet.pipeline import SoftPipeline

  pipe = SoftPipeline.load("fraud_v1.softpipe")   # brak y, brak trenowania
  y_pred = pipe.predict(X_new)                     # → [0, 0, 1, 0, ...]
  proba  = pipe.predict_proba(X_new)[:, 1]         # → [0.02, 0.01, 0.87, ...]

Pliki (gotowe do wdrożenia):
""")
print(f"  {MODEL_DIR}/")
for f in sorted(MODEL_DIR.iterdir()):
    size_kb = f.stat().st_size // 1024
    print(f"    {f.name:<30} {size_kb:>6} KB")
print(f"\n  {MODEL_ZIP.name:<34} {MODEL_ZIP.stat().st_size // 1024:>6} KB  ← to wgraj na serwer")
