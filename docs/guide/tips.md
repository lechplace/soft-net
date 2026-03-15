# Training Tips

Praktyczne wskazówki — kiedy co ustawiać.

---

## Normalizacja danych

soft-net **nie normalizuje automatycznie** — rób to zawsze przed `fit()`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   # ← transform, NIE fit_transform
```

Alternatywnie w Pipeline:

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), SoftClassifier())
```

---

## Dobór presetu

| Zbiór | Preset |
|---|---|
| < 1 000 próbek | `tiny` lub `small` |
| 1k – 10k | `small` lub `medium` |
| 10k – 100k | `medium` lub `large` |
| > 100k | `large_bn` lub `deep` |
| Niezbalansowany (fraud) | `deep` z `dropout ≥ 0.3` |
| Nienormalizowane dane | preset z `batch_norm=True` (`*_bn`) |

---

## Early stopping

Domyślnie włączony (`early_stopping=True`, `patience=10`).
Przy małych danych warto zmniejszyć `patience`:

```python
clf = SoftClassifier(patience=5)
```

Przy dużych danych i długim treningu — zwiększyć:

```python
clf = SoftClassifier(epochs=500, patience=30)
```

---

## Batch size a stabilność

- Mały batch (`16–32`) → większy szum → lepsza generalizacja, wolniej
- Duży batch (`128–512`) → szybszy trening, może gorzej generalizować

```python
# szybki trening na dużym zbiorze
reg = SoftRegressor.from_preset("large", batch_size=256, epochs=50)
```

---

## Verbose

| `verbose` | Co widzisz |
|---|---|
| `0` | nic |
| `1` | info soft-net + pasek postępu Keras |
| `2` | jedna linia na epokę |

```python
clf = SoftClassifier(verbose=1)
clf.fit(X_train, y_train)
# [soft-net] Detected task: binary classification
# [soft-net] loss=binary_crossentropy, activation=sigmoid
# Epoch 1/50 ...
```

---

## Inspect treningu

```python
import pandas as pd

# historia jako DataFrame
hist = pd.DataFrame(clf.history_.history)
print(hist.tail())

# epoka z najlepszym val_loss
best_epoch = hist["val_loss"].idxmin()
print(f"Najlepsza epoka: {best_epoch}, val_loss: {hist['val_loss'][best_epoch]:.4f}")
```
