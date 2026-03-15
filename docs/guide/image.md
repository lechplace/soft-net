# Image Classification

Transfer learning z backbonami SOTA — EfficientNet, MobileNet, ResNet i innymi.

---

## Quick Start

```python
from softnet.image import SoftImageClassifier, BackboneRegistry

# dostępne backbony
BackboneRegistry.list()

clf = SoftImageClassifier(
    num_classes=10,
    backbone="efficientnet_b0",
    epochs=20,
    verbose=1,
)
clf.fit(train_dataset, validation_data=val_dataset)
```

---

## Dostępne backbony

```python
from softnet.image import BackboneRegistry
BackboneRegistry.list()
```

| Backbone | Rodzina | Input size | Charakterystyka |
|---|---|---|---|
| `efficientnet_b0` | EfficientNet | 224×224 | Dobry balans dokładność/szybkość |
| `efficientnet_b3` | EfficientNet | 300×300 | Wyższa dokładność |
| `mobilenet_v2` | MobileNet | 224×224 | Lekki, szybki inference |
| `mobilenet_v3_small` | MobileNet | 224×224 | Najlżejszy |
| `resnet50` | ResNet | 224×224 | Klasyczny, stabilny trening |
| `resnet101` | ResNet | 224×224 | Głębszy ResNet |

---

## Fine-tuning

Dwuetapowy trening: najpierw z zamrożonym backbone, potem odblokowanie warstw.

```python
# Etap 1: trening głowy (backbone zamrożony)
clf = SoftImageClassifier(
    num_classes=5,
    backbone="efficientnet_b0",
    freeze_backbone=True,
    epochs=20,
)
clf.fit(train_ds, validation_data=val_ds)

# Etap 2: fine-tuning — odblokuj ostatnie 30 warstw
clf.fine_tune(train_ds, layers_to_unfreeze=30, epochs=10)
```

---

## Przygotowanie danych

soft-net oczekuje `tf.data.Dataset` lub `keras.utils.image_dataset_from_directory`.

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/val",
    image_size=(224, 224),
    batch_size=32,
)

clf.fit(train_ds, validation_data=val_ds)
```
