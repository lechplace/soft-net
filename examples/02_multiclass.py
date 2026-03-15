"""
Example 2: Multiclass classification — iris / digits dataset.

Demonstrates automatic softmax + sparse_categorical_crossentropy selection.

Run:
    uv run python examples/02_multiclass.py
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from softnet import SoftClassifier

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# custom architecture — just specify layer sizes
clf = SoftClassifier(layers=[256, 128, 64], dropout=0.3, verbose=1)
clf.fit(X_train, y_train)

print(f"\nAccuracy on 10-class digits: {clf.score(X_test, y_test):.4f}")
print("\n--- Why these defaults? ---")
print(clf.explain())
