"""
SoftRegressor — sklearn-compatible neural regressor with smart defaults.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from softnet.base import SoftEstimator, build_mlp
from softnet.inference import ModelConfig


_DEFAULT_LAYERS = [128, 64]


class SoftRegressor(RegressorMixin, SoftEstimator):
    """
    Neural network regressor that automatically configures itself based on y.

    Single or multi-output regression — detected automatically from y shape.
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

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))

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
        if raw.shape[1] == 1:
            return raw.ravel()
        return raw
