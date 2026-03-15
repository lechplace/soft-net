"""
Example 1: Binary classification — breast cancer dataset.

Demonstrates that SoftClassifier requires zero configuration:
- automatically detects binary task
- sets sigmoid + binary_crossentropy
- explain() shows why

Run:
    uv run python examples/01_binary_classification.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from softnet import SoftClassifier

# ── data ──────────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── model — zero configuration needed ─────────────────────────────────
clf = SoftClassifier(verbose=1)
clf.fit(X_train, y_train)

# ── results ────────────────────────────────────────────────────────────
print(f"\nAccuracy: {clf.score(X_test, y_test):.4f}")
print("\n--- Why these defaults? ---")
print(clf.explain())
