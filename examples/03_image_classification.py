"""
Example 3: Image classification with SOTA backbone.

Uses EfficientNetB0 pretrained on ImageNet.
Phase 1: train only the head (backbone frozen)
Phase 2: fine-tune top 20 backbone layers

Run:
    uv run python examples/03_image_classification.py
"""

import numpy as np
import tensorflow as tf
from softnet.image import SoftImageClassifier, BackboneRegistry

# ── show available backbones ──────────────────────────────────────────
print("Available backbones:")
for family in BackboneRegistry.families():
    names = BackboneRegistry.list(family=family)
    print(f"  {family}: {names}")

# ── synthetic dataset (replace with real tf.data.Dataset) ─────────────
def make_fake_dataset(n_samples=64, img_size=224, n_classes=5):
    images = np.random.rand(n_samples, img_size, img_size, 3).astype("float32")
    labels = np.random.randint(0, n_classes, n_samples)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    return ds.batch(16)

train_ds = make_fake_dataset(n_samples=128)
val_ds = make_fake_dataset(n_samples=32)

# ── model ─────────────────────────────────────────────────────────────
clf = SoftImageClassifier(
    num_classes=5,
    backbone="efficientnet_b0",
    head_layers=[256, 128],
    dropout=0.3,
    epochs=3,
    verbose=1,
)

clf.fit(train_ds, validation_data=val_ds)

print("\n--- Configuration ---")
print(clf.explain())

# Phase 2: fine-tune
clf.fine_tune(train_ds, layers_to_unfreeze=20, epochs=2)
print("\nFine-tuning complete.")
