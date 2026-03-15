"""
SoftEstimator — base class for all soft-net estimators.

Inherits from sklearn's BaseEstimator + RegressorMixin/ClassifierMixin pattern.
Wraps a Keras model and exposes sklearn-compatible interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

from softnet.inference import TaskInferrer, ConfigResolver, ModelConfig, TaskInfo


class SoftEstimator(BaseEstimator):
    """
    Abstract base for all soft-net estimators.

    Subclasses must implement:
        _build_model(config: ModelConfig, input_dim: int) -> keras.Model
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
        self.layers = layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        # set after fit()
        self.model_ = None
        self.task_info_: TaskInfo | None = None
        self.config_: ModelConfig | None = None
        self.history_ = None

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftEstimator":
        X, y = check_X_y(X, y, allow_nd=False)

        self.task_info_ = TaskInferrer().infer(y)
        self.config_ = ConfigResolver().resolve(self.task_info_)

        if self.verbose >= 1:
            print(f"[soft-net] Detected task: {self.task_info_}")
            print(f"[soft-net] loss={self.config_.loss}, "
                  f"activation={self.config_.output_activation}")

        self.model_ = self._build_model(self.config_, input_dim=X.shape[1])
        self.history_ = self._train(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        return self._decode_predictions(self.model_.predict(X, verbose=0))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def summary(self) -> None:
        """Print soft-net config + Keras model.summary()."""
        if self.model_ is None:
            print("Model not fitted yet. Call fit() first.")
            return
        print("=" * 60)
        print(f"  soft-net  |  {type(self).__name__}")
        print("=" * 60)
        print(f"  Task           : {self.task_info_}")
        print(f"  Loss           : {self.config_.loss}")
        print(f"  Output activ.  : {self.config_.output_activation}")
        print(f"  Metrics        : {self.config_.metrics}")
        print(f"  Rationale      : {self.config_.rationale}")
        print("-" * 60)
        self.model_.summary()

    def explain(self) -> str:
        """Return human-readable explanation of chosen configuration."""
        if self.config_ is None:
            return "Model not fitted yet."
        return (
            f"Task: {self.task_info_}\n"
            f"Loss: {self.config_.loss}\n"
            f"Output activation: {self.config_.output_activation}\n"
            f"Metrics: {self.config_.metrics}\n"
            f"Rationale: {self.config_.rationale}"
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, config: ModelConfig, input_dim: int):
        raise NotImplementedError

    def _decode_predictions(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _train(self, X: np.ndarray, y: np.ndarray):
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
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
        )

    def _check_is_fitted(self):
        from sklearn.exceptions import NotFittedError
        if self.model_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
