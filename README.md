# soft-net

Framework głębokiego uczenia kompatybilny ze sklearn — inteligentne wartości domyślne, backend Keras.

Przeznaczony dla programistów znających sklearn, którzy chcą korzystać z sieci neuronowych bez konieczności ręcznego dobierania parametrów modelu.

---

## Problem, który rozwiązujemy

Budując sieć neuronową w Keras, trzeba ręcznie wybrać:

```python
# Keras — trzeba wiedzieć, że to są właściwe wybory dla klasyfikacji binarnej
model.compile(
    loss="binary_crossentropy",   # dlaczego nie mse? categorical?
    optimizer=Adam(lr=1e-3),      # dlaczego Adam? jaki lr?
    metrics=["accuracy", "AUC"],  # jakie metryki?
)
output = Dense(1, activation="sigmoid")  # dlaczego sigmoid, nie softmax?
```

soft-net wykrywa typ zadania z danych i dobiera wszystko automatycznie:

```python
# soft-net — zero konfiguracji
clf = SoftClassifier()
clf.fit(X_train, y_train)   # framework sam ustala loss, activation, metrics
print(clf.explain())         # wyjaśnia dlaczego wybrał te wartości
```

---

## Instalacja

Wymagany Python 3.10+.

### Opcja A — bezpośrednio z GitHub (zalecana dla użytkowników)

Nie trzeba klonować repozytorium. Działa z `pip` i `uv`:

```bash
# pip
pip install "git+https://github.com/lechplace/soft-net.git"

# uv
uv pip install "git+https://github.com/lechplace/soft-net.git"

# z obsługą klasyfikacji obrazów
uv pip install "softnet[image] @ git+https://github.com/lechplace/soft-net.git"
```

### Opcja B — nowe środowisko od zera (uv)

```bash
mkdir moj-projekt && cd moj-projekt

uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

uv pip install "git+https://github.com/lechplace/soft-net.git"
```

### Opcja C — dla rozwijających framework (edytowalna)

Zmiany w źródłach są widoczne natychmiast bez reinstalacji:

```bash
git clone https://github.com/lechplace/soft-net.git
cd soft-net

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Weryfikacja instalacji

```python
import softnet
print(softnet.__version__)          # 0.1.0

from softnet import SoftClassifier
from softnet.presets import list_presets
list_presets()
```

### Zależności

| Pakiet | Wersja | Rola |
|--------|--------|------|
| `scikit-learn` | ≥1.3 | API base, metryki, walidacja |
| `keras` | ≥3.0 | backend sieci neuronowych |
| `tensorflow` | ≥2.15 | silnik obliczeń |
| `numpy` | ≥1.24 | operacje na tablicach |

---

## Szybki start

### Klasyfikacja binarna

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from softnet import SoftClassifier

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SoftClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

Framework automatycznie wykrywa, że `y` zawiera dwie klasy i ustawia:
- `loss = binary_crossentropy`
- `output activation = sigmoid`
- `metrics = [accuracy, AUC]`

```python
print(clf.explain())
# Task: BINARY(classes=2)
# Loss: binary_crossentropy
# Output activation: sigmoid
# Rationale: Binary classification: sigmoid squashes output to [0,1];
#            binary_crossentropy is the canonical loss for two-class problems.
```

### Klasyfikacja wieloklasowa

```python
from softnet import SoftClassifier

clf = SoftClassifier(layers=[256, 128, 64], dropout=0.3)
clf.fit(X_train, y_train)   # y zawiera 0,1,2,...,N
```

Automatycznie: `softmax` + `sparse_categorical_crossentropy`.
Nie trzeba one-hot encodować etykiet.

### Regresja

```python
from softnet import SoftRegressor

reg = SoftRegressor(layers=[128, 64], epochs=100)
reg.fit(X_train, y_train)   # y to wartości ciągłe
print(reg.score(X_test, y_test))   # R² score
```

Automatycznie: `linear activation` + `mean_squared_error`.

### Klasyfikacja obrazów (transfer learning)

```python
from softnet.image import SoftImageClassifier, BackboneRegistry

# sprawdź dostępne backbone'y
print(BackboneRegistry.list())
# ['convnext_base', 'convnext_tiny', 'efficientnet_b0', 'efficientnet_b3', ...]

clf = SoftImageClassifier(
    num_classes=10,
    backbone="efficientnet_b0",   # pretrenowany na ImageNet
    head_layers=[256],
    dropout=0.3,
    epochs=20,
)
clf.fit(train_dataset, validation_data=val_dataset)

# fine-tuning — odblokuj ostatnie warstwy backbone'u
clf.fine_tune(train_dataset, layers_to_unfreeze=20)
```

---

## Tabela inteligentnych wartości domyślnych

Framework wykrywa typ zadania na podstawie kształtu i wartości `y`:

| Dane `y` | Wykryte zadanie | Loss | Aktywacja wyjścia | Metryki |
|----------|----------------|------|-------------------|---------|
| `[0, 1, 0, 1]` | Klasyfikacja binarna | `binary_crossentropy` | `sigmoid` | accuracy, AUC |
| `[0, 1, 2, 1, 0]` | Klasyfikacja wieloklasowa | `sparse_categorical_crossentropy` | `softmax` | accuracy |
| `[[0,1],[1,0],[1,1]]` | Multilabel | `binary_crossentropy` | `sigmoid` | accuracy |
| `[1.2, 3.4, 5.6]` | Regresja | `mean_squared_error` | `linear` | MAE |
| `[[1.2, 0.3], [4.5, 2.1]]` | Regresja wielowyjściowa | `mean_squared_error` | `linear` | MAE |

---

## Presety MLP

Gotowe architektury sieci z rozsądnymi wartościami dropout i batch norm.

### Użycie presetu

```python
from softnet import SoftClassifier
from softnet.presets import list_presets

# podejrzyj dostępne presety
list_presets()
# tiny         [64, 32]                     dropout=0.0  bn=False
# small        [128, 64, 32]                dropout=0.1  bn=False
# small_bn     [128, 64, 32]                dropout=0.2  bn=True
# medium       [256, 128, 64, 32]           dropout=0.2  bn=False
# medium_bn    [256, 128, 64, 32]           dropout=0.3  bn=True
# large        [512, 256, 128, 64, 32]      dropout=0.3  bn=False
# large_bn     [512, 256, 128, 64, 32]      dropout=0.3  bn=True
# deep         [512, 256, 256, 128, 64, 32] dropout=0.3  bn=True
# deep_wide    [1024, 512, 256, 128, 64, 32] dropout=0.4 bn=True
# wide         [1024, 512, 256]             dropout=0.4  bn=True
# residual_like [256, 256, 256, 256]        dropout=0.2  bn=True

# utwórz klasyfikator z presetu
clf = SoftClassifier.from_preset("medium")
clf.fit(X_train, y_train)

# nadpisz dowolny parametr
clf = SoftClassifier.from_preset("deep", epochs=200, batch_size=64, verbose=1)
```

### Własny preset w TOML

Utwórz plik `my_presets.toml` i dodaj własne architektury:

```toml
[fraud_net]
layers      = [512, 256, 128, 64]
dropout     = 0.4
batch_norm  = true
description = "Specjalizowana sieć do wykrywania fraudów."

[quick_test]
layers      = [32, 16]
dropout     = 0.0
batch_norm  = false
description = "Minimalna sieć do szybkich testów."
```

Wczytaj i użyj:

```python
from softnet.presets import load_presets_from_toml, list_presets

load_presets_from_toml("my_presets.toml")

list_presets()  # fraud_net i quick_test są teraz dostępne

clf = SoftClassifier.from_preset("fraud_net", epochs=100)
clf.fit(X_train, y_train)
```

### Modyfikacja wbudowanych presetów

Wbudowane presety znajdziesz w pliku `softnet/presets.toml` w repozytorium.
Możesz je edytować bezpośrednio — zmiany są widoczne po ponownym imporcie modułu.

```toml
# softnet/presets.toml  ← edytuj ten plik aby zmienić wbudowane presety

[medium]
layers      = [512, 256, 128, 64]   # zmieniono z [256, 128, 64, 32]
dropout     = 0.25
batch_norm  = false
description = "Mój zmodyfikowany medium."
```

---

## Workflow presets — gotowe cykle ML

`SoftWorkflow` łączy kroki ML (split, scale, fit, validate, save, …) w jeden obiekt
z automatycznym zarządzaniem danymi między krokami.

### Dostępne presety

```python
from softnet.workflows import list_workflows

list_workflows()
```

```
Preset             Kroki
----------------------------------------------------------------------
basic              split → fit → validate
scaled             split → scale → fit → validate
scaled_save        split → scale → fit → validate → save
grid_search        split → scale → grid_search → validate
randomized_search  split → scale → randomized_search → validate
voting             split → scale → voting → validate
full               split → scale → fit → validate → save
full_search        split → scale → grid_search → validate → save
```

### Użycie podstawowe

```python
from softnet.workflows import SoftWorkflow
from softnet import SoftClassifier

wf = SoftWorkflow.from_preset("scaled")
result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))

print(result.score)    # 0.9733  (accuracy lub R²)
print(result.report)   # classification_report / R²/MAE/RMSE
```

### Grid search

```python
wf = SoftWorkflow.from_preset("grid_search")
result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))

print(result.best_params)   # {"epochs": 100, "dropout": 0.2}
print(result.score)
```

Własna siatka parametrów:

```python
from softnet.workflows import SoftWorkflow, GridSearchStep

wf = SoftWorkflow.from_preset(
    "grid_search",
    step_overrides={
        "grid_search": GridSearchStep(
            param_grid={"epochs": [50, 100, 200], "dropout": [0.1, 0.3]},
            cv=3,
        )
    },
)
result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
```

### Voting ensemble

Domyślnie tworzy ansambl trzech modeli: `small`, `medium`, `large`.

```python
wf = SoftWorkflow.from_preset("voting")
result = wf.run(X, y)   # estimator nie jest potrzebny

print(result.score)
```

Własne modele głosujące:

```python
from softnet.workflows import SoftWorkflow, VotingStep
from softnet import SoftClassifier

wf = SoftWorkflow.from_preset(
    "voting",
    step_overrides={
        "voting": VotingStep(
            estimators=[
                ("net_a", SoftClassifier.from_preset("medium", epochs=100)),
                ("net_b", SoftClassifier.from_preset("large",  epochs=100)),
            ],
            voting="soft",
        )
    },
)
result = wf.run(X, y)
```

### Zapis modelu

```python
from softnet.workflows import SoftWorkflow, SaveStep

wf = SoftWorkflow.from_preset(
    "scaled_save",
    step_overrides={"save": SaveStep(path="models/moj_model", format="keras")},
)
result = wf.run(X, y, estimator=SoftClassifier.from_preset("large"))
print(result.saved_path)   # models/moj_model.keras
```

### Własny workflow w TOML

Utwórz plik `my_workflows.toml`:

```toml
[fraud_full]
description = "Pełny cykl do fraud detection"
steps = ["split", "scale", "grid_search", "validate", "save"]

[step_defaults.split]
test_size = 0.15

[step_defaults.save]
path = "models/fraud_model"
```

Wczytaj:

```python
from softnet.workflows import load_workflows_from_toml, SoftWorkflow

load_workflows_from_toml("my_workflows.toml")
wf = SoftWorkflow.from_preset("fraud_full")
result = wf.run(X, y, estimator=SoftClassifier.from_preset("deep"))
```

### WorkflowResult — co zwraca `run()`

| Pole | Typ | Opis |
|------|-----|------|
| `result.score` | `float` | Główna metryka (accuracy lub R²) |
| `result.report` | `str` | Pełny raport tekstowy |
| `result.model` | estimator | Wytrenowany model |
| `result.scaler` | scaler / None | Wytrenowany scaler |
| `result.best_params` | dict / None | Najlepsze params (grid/random search) |
| `result.saved_path` | str / None | Ścieżka zapisanego modelu |
| `result.ctx` | dict | Pełny kontekst — surowe dane, historia treningu |

---

## Dostępne modele bazowe (BackboneRegistry)

Wszystkie modele pretrenowane na ImageNet, ładowane leniwie (brak narzutu przy imporcie).

| Nazwa | Rodzina | Domyślny rozmiar wejścia |
|-------|---------|--------------------------|
| `efficientnet_b0` ⭐ | EfficientNet | 224×224 |
| `efficientnet_b3` | EfficientNet | 300×300 |
| `efficientnet_b7` | EfficientNet | 600×600 |
| `efficientnetv2_s` | EfficientNetV2 | 384×384 |
| `efficientnetv2_l` | EfficientNetV2 | 480×480 |
| `resnet50` | ResNet | 224×224 |
| `resnet50v2` | ResNet | 224×224 |
| `resnet101v2` | ResNet | 224×224 |
| `mobilenet_v2` | MobileNet | 224×224 |
| `mobilenet_v3_small` | MobileNet | 224×224 |
| `mobilenet_v3_large` | MobileNet | 224×224 |
| `convnext_tiny` | ConvNeXt | 224×224 |
| `convnext_base` | ConvNeXt | 224×224 |
| `xception` | Xception | 299×299 |
| `vgg16` | VGG | 224×224 |

⭐ — domyślny, zalecany do większości zadań

```python
from softnet.image import BackboneRegistry

# filtruj po rodzinie
print(BackboneRegistry.list(family="efficientnet"))

# szczegóły konkretnego modelu
spec = BackboneRegistry.get_spec("convnext_base")
print(spec.default_input_size)   # (224, 224)
```

---

## Parametry estymatorów

### SoftClassifier / SoftRegressor

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `layers` | `[128, 64]` | Liczba neuronów w kolejnych warstwach ukrytych |
| `dropout` | `0.0` | Dropout po każdej warstwie ukrytej |
| `batch_norm` | `False` | Batch Normalization po każdej warstwie |
| `epochs` | `50` | Maksymalna liczba epok |
| `batch_size` | `32` | Rozmiar batcha |
| `validation_split` | `0.1` | Część danych treningowych jako walidacja |
| `early_stopping` | `True` | Zatrzymanie gdy walidacja przestaje się poprawiać |
| `patience` | `10` | Liczba epok bez poprawy przed zatrzymaniem |
| `verbose` | `0` | `0` = cicho, `1` = postęp, `2` = pełne logi |

### SoftImageClassifier

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `num_classes` | wymagany | Liczba klas |
| `backbone` | `"efficientnet_b0"` | Nazwa modelu bazowego |
| `backbone_weights` | `"imagenet"` | Wagi pretrenowane (`None` = losowe) |
| `freeze_backbone` | `True` | Zamrożenie backbone podczas treningu head |
| `head_layers` | `[256]` | Warstwy Dense na wyjściu backbone'u |
| `dropout` | `0.3` | Dropout w head |
| `global_pooling` | `"avg"` | `"avg"` lub `"max"` pooling po backbone |
| `epochs` | `20` | Epoki treningu |
| `learning_rate` | `1e-3` | LR podczas treningu head |
| `fine_tune_lr` | `1e-5` | LR podczas fine-tuningu |

---

## Uruchamianie przykładów

```bash
# klasyfikacja binarna (breast cancer)
uv run python examples/01_binary_classification.py

# klasyfikacja wieloklasowa (digits, 10 klas)
uv run python examples/02_multiclass.py

# klasyfikacja obrazów z EfficientNetB0
uv run python examples/03_image_classification.py

# przegląd dostępnych backbone'ów
uv run python examples/04_backbone_registry.py
```

---

## Testy

```bash
# uruchom wszystkie testy
uv run pytest

# z pokryciem kodu
uv run pytest --cov=softnet

# tylko testy reguł (bez Keras)
uv run pytest tests/test_task_inference.py tests/test_config_resolver.py -v
```

---

## Struktura projektu

```
soft-net/
├── softnet/
│   ├── inference/
│   │   ├── task.py        # TaskInferrer — wykrywa typ zadania z y
│   │   └── resolver.py    # ConfigResolver — reguły: zadanie → konfiguracja
│   ├── base/
│   │   ├── estimator.py   # SoftEstimator — baza sklearn + Keras
│   │   └── config.py      # build_mlp() — budowanie warstw Dense
│   ├── tabular/
│   │   ├── classifier.py  # SoftClassifier
│   │   └── regressor.py   # SoftRegressor
│   └── image/
│       ├── classifier.py  # SoftImageClassifier
│       └── backbones.py   # BackboneRegistry
├── examples/
├── tests/
└── pyproject.toml
```

---

## Roadmap

- [ ] `SoftPipeline` — integracja z `sklearn.Pipeline`
- [ ] `SoftGridSearchCV` — hyperparameter search kompatybilny ze sklearn
- [ ] NLP backbone registry (BERT, DistilBERT via Keras-NLP)
- [ ] Auto-preprocessing (normalizacja, encoding kategorycznych cech)
- [ ] Eksport modelu: ONNX, TFLite, SavedModel
