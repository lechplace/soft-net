# Tabular Data

Przewodnik po `SoftClassifier` i `SoftRegressor` — dane tabelaryczne.

---

## SoftClassifier

### Binarna klasyfikacja

```python
from softnet import SoftClassifier

clf = SoftClassifier(
    layers=[128, 64],
    dropout=0.3,
    epochs=50,
    verbose=1,
)
clf.fit(X_train, y_train)
```

!!! note "Auto-config dla binarnej klasyfikacji"
    Gdy `y` zawiera 2 unikalne wartości:

    | Parametr | Wartość |
    |---|---|
    | loss | `binary_crossentropy` |
    | output activation | `sigmoid` |
    | output units | `1` |
    | metrics | `["accuracy"]` |

### Wieloklasowa klasyfikacja

```python
clf = SoftClassifier(epochs=100)
clf.fit(X_train, y_train)   # y może być int lub string
```

!!! note "Auto-config dla multiclass"
    Gdy `y` zawiera ≥ 3 unikalne wartości:

    | Parametr | Wartość |
    |---|---|
    | loss | `sparse_categorical_crossentropy` |
    | output activation | `softmax` |
    | output units | `n_classes` |
    | metrics | `["accuracy"]` |

### predict_proba

```python
proba = clf.predict_proba(X_test)
# binary  → shape (n, 2): [P(class_0), P(class_1)]
# multi   → shape (n, n_classes)

# 5 najbardziej pewnych predykcji
top5 = proba.max(axis=1).argsort()[-5:]
```

### Pipeline sklearn

soft-net jest w pełni kompatybilny z `sklearn.pipeline.Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from softnet import SoftClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    SoftClassifier(epochs=50)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from softnet import SoftClassifier

pipe = make_pipeline(StandardScaler(), SoftClassifier(epochs=30))
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

---

## SoftRegressor

```python
from softnet import SoftRegressor

reg = SoftRegressor(
    layers=[256, 128, 64],
    dropout=0.2,
    epochs=100,
    batch_norm=True,
)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))   # R²
```

!!! note "Auto-config dla regresji"
    Gdy `y` zawiera ciągłe wartości float:

    | Parametr | Wartość |
    |---|---|
    | loss | `mse` |
    | output activation | `linear` |
    | output units | `1` lub `n_outputs` |
    | metrics | `["mae"]` |

### Multi-output regresja

```python
# y shape (n_samples, 3) → automatycznie multi-output
reg.fit(X_train, y_train_multioutput)
```

---

## Wspólne metody

### explain()

```python
print(clf.explain())
```
```
Task: multiclass (3 classes)
Loss: sparse_categorical_crossentropy
Output activation: softmax
Metrics: ['accuracy']
Rationale: 3 unique int labels → multiclass classification
```

### summary()

```python
clf.summary()
```
```
============================================================
  soft-net  |  SoftClassifier
============================================================
  Task           : multiclass (3 classes)
  Loss           : sparse_categorical_crossentropy
  Output activ.  : softmax
  Metrics        : ['accuracy']
  Rationale      : 3 unique int labels → multiclass classification
------------------------------------------------------------
Model: "sequential"
...
```

### history_

```python
import matplotlib.pyplot as plt

h = clf.history_.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(h["accuracy"], label="train"); axes[1].plot(h["val_accuracy"], label="val")
axes[1].set_title("Accuracy"); axes[1].legend()
plt.tight_layout(); plt.show()
```
