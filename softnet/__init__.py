"""
soft-net — sklearn-compatible deep learning framework with smart defaults.

Quick start:
    from softnet import SoftClassifier, SoftRegressor
    from softnet.image import SoftImageClassifier, BackboneRegistry

    clf = SoftClassifier()           # auto-configures loss, activation, metrics
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    print(clf.explain())             # see why these defaults were chosen
"""

from softnet.tabular import SoftClassifier, SoftRegressor
from softnet.image import SoftImageClassifier, BackboneRegistry
from softnet.inference import TaskInferrer, TaskType
from softnet.presets import (
    MLPPreset,
    list_presets,
    get_preset,
    register_preset,
    load_presets_from_toml,
)

__version__ = "0.1.0"

__all__ = [
    "SoftClassifier",
    "SoftRegressor",
    "SoftImageClassifier",
    "BackboneRegistry",
    "TaskInferrer",
    "TaskType",
    # presets
    "MLPPreset",
    "list_presets",
    "get_preset",
    "register_preset",
    "load_presets_from_toml",
]
