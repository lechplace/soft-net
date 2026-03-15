"""
SoftImageClassifier — transfer learning classifier with SOTA backbone.

Usage:
    from softnet.image import SoftImageClassifier

    # Simplest possible — auto-detects binary/multiclass, uses EfficientNetB0
    clf = SoftImageClassifier(num_classes=10, backbone="efficientnet_b0")
    clf.fit(train_ds)
    clf.predict(test_ds)
    print(clf.explain())

    # Fine-tune after initial training
    clf.fine_tune(train_ds, layers_to_unfreeze=20)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from softnet.image.backbones import BackboneRegistry
from softnet.inference import ConfigResolver, TaskInfo, TaskType, ModelConfig


class SoftImageClassifier(BaseEstimator):
    """
    Transfer learning image classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 → binary, >2 → multiclass).
    backbone : str, default "efficientnet_b0"
        Pretrained backbone name. See BackboneRegistry.list().
    backbone_weights : str, default "imagenet"
        Pretrained weights. Use None for random init.
    freeze_backbone : bool, default True
        If True, backbone weights are frozen during initial training.
    head_layers : list[int], default [256]
        Dense layers added on top of backbone.
    dropout : float, default 0.3
    global_pooling : str, default "avg"
        Pooling applied to backbone output: "avg" or "max".
    epochs : int, default 20
    batch_size : int, default 32
    learning_rate : float, default 1e-3
    fine_tune_lr : float, default 1e-5
        Learning rate used during fine-tuning phase.
    early_stopping : bool, default True
    patience : int, default 5
    verbose : int, default 0

    Attributes
    ----------
    model_ : keras.Model
    config_ : ModelConfig
    task_info_ : TaskInfo
    history_ : keras.callbacks.History
    """

    def __init__(
        self,
        *,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        backbone_weights: str = "imagenet",
        freeze_backbone: bool = True,
        head_layers: list[int] | None = None,
        dropout: float = 0.3,
        global_pooling: str = "avg",
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        fine_tune_lr: float = 1e-5,
        early_stopping: bool = True,
        patience: int = 5,
        verbose: int = 0,
    ):
        self.num_classes = num_classes
        self.backbone = backbone
        self.backbone_weights = backbone_weights
        self.freeze_backbone = freeze_backbone
        self.head_layers = head_layers
        self.dropout = dropout
        self.global_pooling = global_pooling
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        self.model_ = None
        self.config_: ModelConfig | None = None
        self.task_info_: TaskInfo | None = None
        self.history_ = None
        self._backbone_model = None

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def fit(self, dataset, validation_data=None) -> "SoftImageClassifier":
        self.task_info_ = self._infer_task()
        self.config_ = ConfigResolver().resolve(self.task_info_)

        if self.verbose >= 1:
            spec = BackboneRegistry.get_spec(self.backbone)
            print(f"[soft-net] Backbone: {spec.name} ({spec.family})")
            print(f"[soft-net] Task: {self.task_info_}")
            print(f"[soft-net] loss={self.config_.loss}, activation={self.config_.output_activation}")

        self.model_ = self._build_model()
        self.history_ = self._train(dataset, validation_data)
        return self

    def predict(self, dataset) -> np.ndarray:
        self._check_is_fitted()
        raw = self.model_.predict(dataset, verbose=0)
        return self._decode(raw)

    def predict_proba(self, dataset) -> np.ndarray:
        self._check_is_fitted()
        raw = self.model_.predict(dataset, verbose=0)
        if self.task_info_.task == TaskType.BINARY:
            return np.hstack([1 - raw, raw])
        return raw

    def score(self, dataset, y=None) -> float:
        preds = self.predict(dataset)
        if y is not None:
            return accuracy_score(y, preds)
        raise ValueError("Provide y to compute accuracy on a dataset.")

    def fine_tune(self, dataset, *, layers_to_unfreeze: int = 30, epochs: int | None = None) -> "SoftImageClassifier":
        """
        Unfreeze top N backbone layers and continue training with low LR.
        Call after fit().
        """
        import keras

        self._check_is_fitted()

        backbone = self._backbone_model
        backbone.trainable = True

        # freeze all but the last N layers
        for layer in backbone.layers[:-layers_to_unfreeze]:
            layer.trainable = False

        if self.verbose >= 1:
            trainable = sum(1 for l in backbone.layers if l.trainable)
            print(f"[soft-net] Fine-tuning: {trainable} trainable layers in backbone")

        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.fine_tune_lr),
            loss=self.config_.loss,
            metrics=self.config_.metrics,
        )

        ft_epochs = epochs or max(self.epochs // 2, 5)
        self.model_.fit(
            dataset,
            epochs=ft_epochs,
            verbose=self.verbose,
        )
        return self

    def explain(self) -> str:
        if self.config_ is None:
            return "Model not fitted yet."
        spec = BackboneRegistry.get_spec(self.backbone)
        return (
            f"Backbone: {spec.name} (family={spec.family}, "
            f"default_input={spec.default_input_size})\n"
            f"Task: {self.task_info_}\n"
            f"Loss: {self.config_.loss}\n"
            f"Output activation: {self.config_.output_activation}\n"
            f"Rationale: {self.config_.rationale}"
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _infer_task(self) -> TaskInfo:
        n = self.num_classes
        if n == 2:
            return TaskInfo(task=TaskType.BINARY, n_outputs=1, n_classes=2)
        return TaskInfo(task=TaskType.MULTICLASS, n_outputs=n, n_classes=n)

    def _build_model(self):
        import keras
        from keras import layers

        spec = BackboneRegistry.get_spec(self.backbone)
        h, w = spec.default_input_size

        self._backbone_model = spec.factory(
            weights=self.backbone_weights,
            include_top=False,
            input_shape=(h, w, 3),
        )
        self._backbone_model.trainable = not self.freeze_backbone

        # ── head ──────────────────────────────────────────────────────
        pooling = (
            layers.GlobalAveragePooling2D()
            if self.global_pooling == "avg"
            else layers.GlobalMaxPooling2D()
        )

        x = self._backbone_model.output
        x = pooling(x)

        for units in (self.head_layers or [256]):
            x = layers.Dense(units, activation="relu")(x)
            if self.dropout > 0.0:
                x = layers.Dropout(self.dropout)(x)

        output = layers.Dense(
            self.config_.output_units,
            activation=self.config_.output_activation,
        )(x)

        model = keras.Model(inputs=self._backbone_model.input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.config_.loss,
            metrics=self.config_.metrics,
        )
        return model

    def _train(self, dataset, validation_data):
        from keras.callbacks import EarlyStopping

        callbacks = []
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )

        return self.model_.fit(
            dataset,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose,
        )

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        if self.task_info_.task == TaskType.BINARY:
            return (raw.ravel() >= 0.5).astype(int)
        return np.argmax(raw, axis=1)

    def _check_is_fitted(self):
        from sklearn.exceptions import NotFittedError
        if self.model_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted. Call fit() first."
            )
