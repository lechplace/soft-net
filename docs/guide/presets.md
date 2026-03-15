# MLP Presets

Gotowe architektury sieci — wybierz preset zamiast ręcznie ustawiać `layers`, `dropout`, `batch_norm`.

---

## Dostępne presety

```python
from softnet import list_presets
list_presets()
```

| Preset | Warstwy | Dropout | BatchNorm | Kiedy użyć |
|---|---|---|---|---|
| `tiny` | [64, 32] | 0.0 | ✗ | Prototypowanie, bardzo małe dane |
| `small` | [128, 64, 32] | 0.1 | ✗ | Proste zadania (Iris, <1k próbek) |
| `small_bn` | [128, 64, 32] | 0.2 | ✓ | Jak small + szybsza zbieżność |
| `medium` | [256, 128, 64, 32] | 0.2 | ✗ | Ogólne zastosowania — dobry punkt startowy |
| `medium_bn` | [256, 128, 64, 32] | 0.3 | ✓ | Medium + odporny na overfitting |
| `large` | [512, 256, 128, 64, 32] | 0.3 | ✗ | Średnie zbiory (10k–100k) |
| `large_bn` | [512, 256, 128, 64, 32] | 0.3 | ✓ | Large + nienormalizowane dane |
| `deep` | [512, 256, 256, 128, 64, 32] | 0.3 | ✓ | Złożone zadania, fraud detection |
| `deep_wide` | [1024, 512, 256, 128, 64, 32] | 0.4 | ✓ | Maksymalna pojemność, duże zbiory |
| `wide` | [1024, 512, 256] | 0.4 | ✓ | Informatywne features, płytka sieć |
| `residual_like` | [256, 256, 256, 256] | 0.2 | ✓ | Stabilny trening, styl ResNet |

---

## Użycie

### from_preset()

```python
from softnet import SoftClassifier, SoftRegressor

# classifier z presetem — epochs i inne parametry nadpisujesz przez kwargs
clf = SoftClassifier.from_preset("medium", epochs=100, verbose=1)
clf.fit(X_train, y_train)

# regressor
reg = SoftRegressor.from_preset("large_bn", epochs=200, batch_size=64)
reg.fit(X_train, y_train)
```

### Podejrzenie presetu

```python
from softnet import get_preset

p = get_preset("medium")
print(p)
# MLPPreset(name='medium', layers=[256, 128, 64, 32], dropout=0.2, batch_norm=False)

print(p.layers)    # [256, 128, 64, 32]
print(p.dropout)   # 0.2
```

---

## Własne presety

### Przez Python (tymczasowe — trwają do restartu)

```python
from softnet import MLPPreset, register_preset, SoftClassifier

register_preset(MLPPreset(
    name="fraud_net",
    layers=[512, 256, 128, 64],
    dropout=0.35,
    batch_norm=True,
    description="Zoptymalizowany dla niezbalansowanych danych",
))

clf = SoftClassifier.from_preset("fraud_net", epochs=50)
```

### Przez plik TOML (trwałe)

Utwórz plik np. `~/my_presets.toml`:

```toml
[fraud_net]
layers      = [512, 256, 128, 64]
dropout     = 0.35
batch_norm  = true
description = "Zoptymalizowany dla niezbalansowanych danych"

[quick_test]
layers      = [32, 16]
dropout     = 0.0
batch_norm  = false
description = "Minimalna sieć do szybkiego testowania pipeline"
```

Wczytaj i używaj:

```python
from softnet import load_presets_from_toml, SoftClassifier, list_presets

load_presets_from_toml("~/my_presets.toml")
list_presets()   # fraud_net i quick_test są teraz widoczne

clf = SoftClassifier.from_preset("fraud_net", epochs=30)
```

!!! tip "Wczytywanie przy starcie projektu"
    Dodaj `load_presets_from_toml(...)` na początku notebooka lub
    w `conftest.py` — presety będą dostępne przez cały czas pracy.

---

## Edycja wbudowanych presetów

Wbudowane presety siedzą w pliku:

```
softnet/presets.toml
```

Otwórz go w edytorze, zmień wartości, zrestartuj kernel — zmiany są od razu aktywne.
Nie trzeba przeinstalowywać pakietu (instalacja edytowalna).
