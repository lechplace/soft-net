"""
SoftClassifier — sklearn-compatible neural classifier with smart defaults.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from softnet.base import SoftEstimator, build_mlp
from softnet.inference import ModelConfig, TaskType


_DEFAULT_LAYERS = [128, 64]


class SoftClassifier(ClassifierMixin, SoftEstimator):
    """
    Neural network classifier with automatic configuration.

    Detects the classification task type from ``y`` and sets loss,
    output activation and metrics without any manual configuration:

    +------------------+------------------------------+-----------+
    | Task             | Loss                         | Activation|
    +==================+==============================+===========+
    | Binary           | binary_crossentropy          | sigmoid   |
    +------------------+------------------------------+-----------+
    | Multiclass       | sparse_categorical_crossentr | softmax   |
    +------------------+------------------------------+-----------+

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
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit()``.

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
    **Binary classification (Credit Fraud):**

    >>> from softnet import SoftClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> # X, y — loaded from any source (e.g. kaggle credit card fraud)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> scaler = StandardScaler()
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_test  = scaler.transform(X_test)
    >>>
    >>> clf = SoftClassifier(layers=[128, 64], dropout=0.3, epochs=30)
    >>> clf.fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.9994...

    **Multiclass classification (Iris):**

    >>> from softnet import SoftClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>>
    >>> clf = SoftClassifier(epochs=100, verbose=1)
    >>> clf.fit(X_train, y_train)
    [soft-net] Detected task: multiclass (3 classes)
    [soft-net] loss=sparse_categorical_crossentropy, activation=softmax
    >>> clf.score(X_test, y_test)
    0.9736...

    See Also
    --------
    SoftRegressor : For continuous target prediction.
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
        self._label_encoder: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftClassifier":
        """
        Fit the classifier to training data.

        Encodes class labels via ``LabelEncoder``, detects the task type,
        builds and trains the Keras model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Should be scaled (e.g. ``StandardScaler``).

        y : array-like of shape (n_samples,)
            Class labels. Accepts integers, strings, or floats.
            Binary (2 unique values) or multiclass (3+) detected automatically.

        Returns
        -------
        self : SoftClassifier
            Fitted estimator.

        Examples
        --------
        >>> clf = SoftClassifier(layers=[64, 32], epochs=50, dropout=0.2)
        >>> clf.fit(X_train, y_train)
        SoftClassifier(dropout=0.2, epochs=50, layers=[64, 32])
        """
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        return super().fit(X, y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in ``X``.

        For binary classification returns shape ``(n_samples, 2)``
        — columns are ``[P(class_0), P(class_1)]``.
        For multiclass returns shape ``(n_samples, n_classes)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Each row sums to 1.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit()``.

        Examples
        --------
        >>> proba = clf.predict_proba(X_test)
        >>> proba[:3]
        array([[0.97, 0.03],
               [0.12, 0.88],
               [0.55, 0.45]])
        >>> # most confident predictions:
        >>> proba.max(axis=1).argsort()[-5:]
        """
        self._check_is_fitted()
        X = check_array(X)
        raw = self.model_.predict(X, verbose=0)
        if self.task_info_.task == TaskType.BINARY:
            return np.hstack([1 - raw, raw])
        return raw

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples. Range: ``[0.0, 1.0]``.

        Examples
        --------
        >>> clf.fit(X_train, y_train)
        >>> clf.score(X_test, y_test)
        0.9736...
        """
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
