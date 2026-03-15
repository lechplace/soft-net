# soft-net

**sklearn-compatible deep learning framework with smart defaults.**

soft-net rozszerza sklearn o sieci neuronowe (Keras backend) — bez konieczności ręcznego konfigurowania loss functions, aktywacji czy metryk. Framework sam wykrywa typ zadania i dobiera właściwe ustawienia.

---

## Dlaczego soft-net?

=== "Bez soft-net"

    ```python
    import keras

    # musisz wiedzieć...
    model = keras.Sequential([...])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # ← skąd to wiedzieć?
        metrics=["accuracy"],                    # ← i to?
    )
    model.fit(X, y, epochs=50, validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    ```

=== "Z soft-net"

    ```python
    from softnet import SoftClassifier

    clf = SoftClassifier()
    clf.fit(X, y)   # ← loss, activation, metrics — wykryte automatycznie
    ```

---

## Instalacja

```bash
pip install soft-net
```

!!! tip "Opcjonalnie — klasyfikacja obrazów"
    ```bash
    pip install "soft-net[image]"
    ```

---

## 5 minut — pierwszy model

```python
from softnet import SoftClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dane
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = StandardScaler().fit_transform(X_train)

# model — zero konfiguracji
clf = SoftClassifier(verbose=1)
clf.fit(X_train, y_train)
# [soft-net] Detected task: multiclass (3 classes)
# [soft-net] loss=sparse_categorical_crossentropy, activation=softmax

print(clf.score(X_test, y_test))   # ~0.97
clf.summary()                       # architektura + config
```

---

## Kluczowe funkcje

| Funkcja | Opis |
|---|---|
| **Auto-config** | loss, activation, metrics dobierane automatycznie z `y` |
| **sklearn API** | `fit / predict / score / predict_proba` — jak zawsze |
| **MLP Presets** | gotowe architektury: `tiny`, `medium`, `deep` i inne |
| **Image** | transfer learning (EfficientNet, MobileNet, ResNet) |
| **Explain** | `clf.explain()` — dlaczego takie ustawienia? |
| **Summary** | `clf.summary()` — config + keras architektura |

---

## Auto-konfiguracja — reguły

| Dane w `y` | Wykryte zadanie | Loss | Aktywacja |
|---|---|---|---|
| 2 unikalne wartości | binary | `binary_crossentropy` | `sigmoid` |
| 3+ unikalne wartości int | multiclass | `sparse_categorical_crossentropy` | `softmax` |
| float ciągłe | regression | `mse` | `linear` |

---

## Nawigacja

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quick Start](guide/quickstart.md)**

    Pierwsze kroki — Iris, Credit Fraud, California Housing

-   :material-table: **[Tabular Data](guide/tabular.md)**

    SoftClassifier i SoftRegressor w praktyce

-   :material-image: **[Image Classification](guide/image.md)**

    Transfer learning z EfficientNet, MobileNet, ResNet

-   :material-layers: **[MLP Presets](guide/presets.md)**

    Gotowe architektury — edycja przez TOML

-   :material-code-tags: **[API Reference](api/classifier.md)**

    Pełna dokumentacja klas i metod

</div>
