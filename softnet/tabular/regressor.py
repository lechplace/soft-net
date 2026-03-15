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
    Neural network regressor with automatic configuration.

    Detects single-output vs multi-output regression from ``y`` shape
    and sets ``mse`` loss and ``linear`` activation automatically.

    Parameters
    ----------
    layers : list of int, default [128, 64]
        Hidden layer sizes.
        Example: ``[256, 128, 64]`` → three hidden layers.

    dropout : float, default 0.0
        Dropout rate after each hidden layer. Range: ``[0.0, 1.0)``.

    batch_norm : bool, default False
        Add BatchNormalization after each hidden layer.

    epochs : int, default 50
        Maximum training epochs (may stop earlier with early stopping).

    batch_size : int, default 32
        Samples per gradient update.

    validation_split : float, default 0.1
        Fraction of data held out as validation during training.

    early_stopping : bool, default True
        Stop training when ``val_loss`` stops improving.

    patience : int, default 10
        Epochs to wait before early stop triggers.

    verbose : int, default 0
        ``0`` silent, ``1`` config + progress bar, ``2`` one line/epoch.

    Attributes
    ----------
    model_ : keras.Model
        Fitted Keras model.

    task_info_ : TaskInfo
        Detected task type and metadata.

    config_ : ModelConfig
        Resolved Keras config (loss, activation, metrics, lr).

    history_ : keras.callbacks.History
        Epoch-by-epoch training metrics.

    Examples
    --------
    **Single-output regression:**

    >>> from softnet import SoftRegressor
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> scaler = StandardScaler()
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_test  = scaler.transform(X_test)
    >>>
    >>> reg = SoftRegressor(layers=[128, 64], dropout=0.2, epochs=100)
    >>> reg.fit(X_train, y_train)
    >>> reg.score(X_test, y_test)       # R²
    0.821...

    **Access training history:**

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(reg.history_.history["loss"], label="train")
    >>> plt.plot(reg.history_.history["val_loss"], label="val")
    >>> plt.legend(); plt.show()

    See Also
    --------
    SoftClassifier : For discrete label prediction.
    SoftEstimator.summary : Print model architecture + config.
    SoftEstimator.explain : Config rationale as a string.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftRegressor":
        """
        Fit the regressor to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Should be scaled (e.g. ``StandardScaler``).

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Continuous target values.
            1D array → single output, 2D array → multi-output.

        Returns
        -------
        self : SoftRegressor
            Fitted estimator.

        Examples
        --------
        >>> reg = SoftRegressor(layers=[64, 32], epochs=50)
        >>> reg.fit(X_train, y_train)
        SoftRegressor(epochs=50, layers=[64, 32])
        """
        return super().fit(X, y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R² (coefficient of determination) on the given test data.

        R² = 1 is perfect prediction. R² = 0 means the model predicts
        the mean. Negative values mean the model is worse than the mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True target values.

        Returns
        -------
        r2 : float
            R² score.

        Examples
        --------
        >>> reg.fit(X_train, y_train)
        >>> reg.score(X_test, y_test)
        0.821...
        """
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
