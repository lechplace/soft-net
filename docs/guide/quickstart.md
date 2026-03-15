# Quick Start

Trzy kompletne przykłady — od wczytania danych do ewaluacji.

---

## Iris — klasyfikacja wieloklasowa

soft-net wykrywa 3 klasy i automatycznie ustawia
`sparse_categorical_crossentropy` + `softmax`.

```python
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from softnet import SoftClassifier

# wczytanie przez kagglehub
path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(f"{path}/Iris.csv")

X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
y = df["Species"].values   # stringi → soft-net koduje automatycznie

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = SoftClassifier(epochs=100, verbose=1)
clf.fit(X_train, y_train)
```

```
[soft-net] Detected task: multiclass (3 classes)
[soft-net] loss=sparse_categorical_crossentropy, activation=softmax
```

```python
print(clf.score(X_test, y_test))   # ~0.97
print(clf.explain())
clf.summary()
```

!!! tip "Preset zamiast ręcznej konfiguracji"
    ```python
    clf = SoftClassifier.from_preset("small", epochs=100)
    ```

---

## Credit Card Fraud — klasyfikacja binarna

soft-net wykrywa 2 klasy i ustawia `binary_crossentropy` + `sigmoid`.

```python
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from softnet import SoftClassifier

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")

X = df.drop(columns=["Time", "Class"]).values
y = df["Class"].values   # 0 / 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# preset "deep" — dobry dla niezbalansowanych danych o dużej liczbie cech
clf = SoftClassifier.from_preset("deep", epochs=30, verbose=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

```
[soft-net] Detected task: binary classification
[soft-net] loss=binary_crossentropy, activation=sigmoid
```

!!! warning "Niezbalansowane dane"
    Credit Fraud jest silnie niezbalansowany (~0.17% fraudów).
    Rozważ `class_weight` lub oversampling przed treningiem.

---

## California Housing — regresja

soft-net wykrywa wartości float i ustawia `mse` + `linear`.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from softnet import SoftRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

reg = SoftRegressor.from_preset("medium", epochs=100, verbose=1)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))   # R²
```

```
[soft-net] Detected task: regression (1 output)
[soft-net] loss=mse, activation=linear
```

```python
# wizualizacja przebiegu treningu
import matplotlib.pyplot as plt

hist = reg.history_.history
plt.plot(hist["loss"],     label="train loss")
plt.plot(hist["val_loss"], label="val loss")
plt.legend(); plt.xlabel("epoch"); plt.tight_layout()
plt.show()
```

---

## Co dalej?

- [Tabular Data Guide](tabular.md) — więcej opcji dla danych tabelarycznych
- [MLP Presets](presets.md) — dostępne architektury i własne presety
- [API Reference](../api/classifier.md) — pełna dokumentacja parametrów
