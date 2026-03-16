"""
Przykład 08 — SoftPipeline: train → save → load → predict (produkcja)
======================================================================

Demonstruje cały cykl produkcyjny:

  DEV:  wf.run(X, y)  → result.to_pipeline() → pipe.save()
  PROD: SoftPipeline.load() → pipe.predict(X_new)   ← brak y, brak trenowania

Dataset: UCI Credit Card Fraud (Kaggle: mlg-ulb/creditcardfraud)
         284 807 transakcji, 492 fraud (0.17%), 30 cech PCA (V1–V28, Amount, Time)

Wymagania:
    pip install kaggle
    # Umieść ~/.kaggle/kaggle.json z tokenem API (https://kaggle.com/settings → API)

Uruchomienie:
    python examples/08_pipeline_save_load.py
"""

# ── 0. importy ────────────────────────────────────────────────────────────────

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from softnet import SoftClassifier, SoftPipeline
from softnet.workflows import SoftWorkflow

DATASET_SLUG = "mlg-ulb/creditcardfraud"
CACHE_DIR    = Path.home() / ".cache" / "soft-net-datasets"
CSV_PATH     = CACHE_DIR / "creditcard.csv"
MODEL_DIR    = Path("/tmp/soft-net-fraud-v1")
MODEL_ZIP    = Path("/tmp/soft-net-fraud-v1.softpipe")

print("=" * 60)
print("  soft-net — Pipeline save / load demo")
print("=" * 60)

# ── 1. Pobieranie danych z Kaggle ─────────────────────────────────────────────

def download_from_kaggle(slug: str, dest: Path) -> None:
    """Pobierz dataset z Kaggle API i wypakuj CSV."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        raise ImportError(
            "Wymagana biblioteka: pip install kaggle\n"
            "Oraz token API w ~/.kaggle/kaggle.json\n"
            "Pobierz go z: https://kaggle.com/settings → API → Create New Token"
        )

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "creditcardfraud.zip"

    print(f"      Pobieranie {slug} …")
    api = KaggleApiExtended()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), quiet=False)

    print(f"      Rozpakowywanie …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)


print(f"\n[1/5] Dataset: Credit Card Fraud ({DATASET_SLUG})")

if CSV_PATH.exists():
    print(f"      Używam z cache: {CSV_PATH}")
else:
    print(f"      Brak cache — pobieram z Kaggle …")
    download_from_kaggle(DATASET_SLUG, CACHE_DIR)
    print(f"      Zapisano → {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
X  = df.drop(columns=["Class"]).values.astype(np.float32)
y  = df["Class"].values.astype(np.int32)
print(f"      Shape: {X.shape}  |  Fraud: {y.sum()} ({y.mean()*100:.2f}%)")

# ── 2. Trenowanie na DEV ──────────────────────────────────────────────────────

print("\n[2/5] Trenowanie workflow (RobustScaler → leaf_encoding → fit) …")

clf = SoftClassifier.from_preset(
    "imb_funnel_heavy",   # [512, 256, 128, 64], dropout=0.4, BN=True
    epochs=30,
    verbose=0,
)

wf = SoftWorkflow.from_preset("imb_leaf_robust")

result = wf.run(X, y, estimator=clf)

print(f"\n      Accuracy : {result.score:.4f}")
if result.ctx.get("validation", {}).get("roc_auc"):
    print(f"      AUC-ROC  : {result.ctx['validation']['roc_auc']:.4f}")
print(f"\n{result.report}")

# ── 3. Eksport do SoftPipeline ────────────────────────────────────────────────

print("[3/5] Eksport do SoftPipeline …")
pipe = result.to_pipeline()
pipe.summary()

pipe.save(MODEL_DIR)
pipe.save(MODEL_DIR, as_zip=True)

print(f"\n      Katalog : {MODEL_DIR}/")
print(f"      Zip     : {MODEL_ZIP}")

# ── 4. Załadowanie pipeline (symulacja PROD) ──────────────────────────────────

print("\n[4/5] Ładowanie pipeline z dysku (symulacja prod) …")

pipe_dir = SoftPipeline.load(MODEL_DIR)
pipe_zip = SoftPipeline.load(MODEL_ZIP)

print(f"      Załadowano (katalog): {pipe_dir}")
print(f"      Załadowano (zip)    : {pipe_zip}")
pipe_zip.summary()

# ── 5. Inferencja na PROD — brak y, brak trenowania ──────────────────────────

print("\n[5/5] Inferencja na nowych danych (brak y) …")

rng   = np.random.default_rng(0)
idx   = rng.choice(len(X), size=500, replace=False)
X_new = X[idx]   # surowe dane — identyczne cechy jak podczas treningu

# predict — klasy
y_pred = pipe_zip.predict(X_new)
counts = np.bincount(y_pred.astype(int), minlength=2)
print(f"\n      predict()             → ok={counts[0]}, fraud={counts[1]}")

# predict_proba — prawdopodobieństwo fraudu
proba = pipe_zip.predict_proba(X_new)
fraud_score = proba[:, 1]
print(f"      predict_proba()[:,1]  → min={fraud_score.min():.3f}, "
      f"max={fraud_score.max():.3f}, mean={fraud_score.mean():.4f}")

# decision_score — surowy output sieci (własny próg)
raw = pipe_zip.decision_score(X_new)
print(f"      decision_score() shape: {raw.shape}")

custom_threshold = 0.3
y_custom = (fraud_score >= custom_threshold).astype(int)
print(f"\n      Próg 0.5 → fraudów wykrytych: {y_pred.sum()}")
print(f"      Próg 0.3 → fraudów wykrytych: {y_custom.sum()}  (mniej missed, więcej FP)")

# ── Podsumowanie ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Cykl produkcyjny — ściąga")
print("=" * 60)
print("""
DEV (trening i eksport):
  wf     = SoftWorkflow.from_preset("imb_leaf_robust")
  clf    = SoftClassifier.from_preset("imb_funnel_heavy", epochs=30)
  result = wf.run(X, y, estimator=clf)

  pipe = result.to_pipeline()
  pipe.save("fraud_v1", as_zip=True)   →  fraud_v1.softpipe

PROD (inferencja — brak y, brak trenowania):
  from softnet.pipeline import SoftPipeline

  pipe   = SoftPipeline.load("fraud_v1.softpipe")
  y_pred = pipe.predict(X_new)           # klasy [0, 1, 0, …]
  proba  = pipe.predict_proba(X_new)[:,1]  # P(fraud)
  raw    = pipe.decision_score(X_new)    # surowy output → własny próg
""")

print("Pliki gotowe do wdrożenia:")
for f in sorted(MODEL_DIR.iterdir()):
    print(f"  {MODEL_DIR.name}/{f.name:<30} {f.stat().st_size // 1024:>6} KB")
print(f"\n  {MODEL_ZIP.name:<42} {MODEL_ZIP.stat().st_size // 1024:>6} KB  ← wgraj na serwer")
