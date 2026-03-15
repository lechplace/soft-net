"""
SoftClassifier — sklearn-compatible neural classifier with smart defaults.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from softnet.base import SoftEstimator, build_mlp
from softnet.inference import ModelConfig, TaskType


_DEFAULT_LAYERS = [128, 64]


class SoftClassifier(ClassifierMixin, SoftEstimator):
    """
    Neural network classifier that automatically configures itself based on y.

    Parameters
    ----------
    layers : list[int], default [128, 64]
    dropout : float, default 0.0
    batch_norm : bool, default False
    epochs : int, default 50
    batch_size : int, default 32
    validation_split : float, default 0.1
    early_stopping : bool, default True
    patience : int, default 10
    verbose : int, default 0

    Attributes
    ----------
    classes_ : ndarray
    task_info_ : TaskInfo
    config_ : ModelConfig
    history_ : keras.callbacks.History
    """

    def __init__(
        self,
        *,
        layers: list[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 0,
    ):
        super().__init__(
            layers=layers,
            dropout=dropout,
            batch_norm=batch_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            verbose=verbose,
        )
        self._label_encoder: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftClassifier":
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        return super().fit(X, y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        raw = self.model_.predict(X, verbose=0)
        if self.task_info_.task == TaskType.BINARY:
            return np.hstack([1 - raw, raw])
        return raw

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def _build_model(self, config: ModelConfig, input_dim: int):
        import keras

        hidden = self.layers if self.layers is not None else _DEFAULT_LAYERS
        model = build_mlp(
            hidden_units=hidden,
            output_units=config.output_units,
            output_activation=config.output_activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            input_dim=input_dim,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=config.loss,
            metrics=config.metrics,
        )
        return model

    def _decode_predictions(self, raw: np.ndarray) -> np.ndarray:
        if self.task_info_.task == TaskType.BINARY:
            encoded = (raw.ravel() >= 0.5).astype(int)
        elif self.task_info_.task == TaskType.MULTICLASS:
            encoded = np.argmax(raw, axis=1)
        else:
            return (raw >= 0.5).astype(int)

        return self._label_encoder.inverse_transform(encoded)
