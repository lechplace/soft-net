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
    Abstract base class for all soft-net estimators.

    Automatically infers the task type from ``y`` (binary, multiclass,
    regression) and configures the Keras model accordingly — loss function,
    output activation, and metrics are chosen without user intervention.

    Subclasses must implement:
        - ``_build_model(config, input_dim)``
        - ``_decode_predictions(raw)``

    Parameters
    ----------
    layers : list of int, default [128, 64]
        Number of units in each hidden Dense layer.
        Example: ``[256, 128, 64]`` creates three hidden layers.

    dropout : float, default 0.0
        Dropout rate applied after each hidden layer.
        Must be in range ``[0.0, 1.0)``.
        Ignored when set to ``0.0``.

    batch_norm : bool, default False
        Whether to insert BatchNormalization after each hidden layer
        (before the activation / dropout).

    epochs : int, default 50
        Maximum number of training epochs.
        With ``early_stopping=True`` training may stop earlier.

    batch_size : int, default 32
        Number of samples per gradient update.

    validation_split : float, default 0.1
        Fraction of training data used as validation set during ``fit``.
        Required when ``early_stopping=True``.

    early_stopping : bool, default True
        Whether to use Keras ``EarlyStopping`` callback.
        Monitors ``val_loss`` with ``restore_best_weights=True``.

    patience : int, default 10
        Number of epochs with no improvement before stopping.
        Only relevant when ``early_stopping=True``.

    verbose : int, default 0
        Verbosity level.
        ``0`` — silent, ``1`` — soft-net config info + Keras progress bar,
        ``2`` — one line per epoch.

    Attributes
    ----------
    model_ : keras.Model
        Fitted Keras model. Available after ``fit()``.

    task_info_ : TaskInfo
        Detected task metadata (type, number of classes, etc.).

    config_ : ModelConfig
        Resolved Keras configuration (loss, activation, metrics, lr).

    history_ : keras.callbacks.History
        Training history returned by ``model.fit()``.

    See Also
    --------
    SoftClassifier : For classification tasks.
    SoftRegressor  : For regression tasks.
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
        """
        Fit the neural network to training data.

        Automatically infers the task type from ``y``, selects the
        appropriate loss function and output activation, builds the
        Keras model, and trains it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
            - Integer or string labels → classification.
            - Float values → regression.

        Returns
        -------
        self : SoftEstimator
            Fitted estimator (for method chaining).

        Examples
        --------
        >>> from softnet import SoftClassifier
        >>> import numpy as np
        >>> X = np.random.rand(200, 10)
        >>> y = np.random.randint(0, 2, 200)
        >>> clf = SoftClassifier(epochs=20)
        >>> clf.fit(X, y)
        SoftClassifier(epochs=20)
        """
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
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples. Must have the same number of features
            as the data passed to ``fit()``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
            For classifiers: original class labels (decoded via LabelEncoder).
            For regressors: continuous values.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit()``.

        Examples
        --------
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> y_pred[:5]
        array([0, 1, 1, 0, 1])
        """
        self._check_is_fitted()
        X = check_array(X)
        return self._decode_predictions(self.model_.predict(X, verbose=0))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the evaluation metric on the given test data.

        Metric depends on the subclass:
        - ``SoftClassifier`` → accuracy
        - ``SoftRegressor``  → R² score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels or values.

        Returns
        -------
        score : float
            Accuracy (classifier) or R² (regressor).

        Examples
        --------
        >>> clf.fit(X_train, y_train)
        >>> clf.score(X_test, y_test)
        0.9533...
        """
        raise NotImplementedError

    def summary(self) -> None:
        """
        Print soft-net configuration and Keras model architecture.

        Combines the automatically resolved training configuration
        (task type, loss, activation, rationale) with the standard
        ``keras.Model.summary()`` output.

        Must be called after ``fit()``.

        Examples
        --------
        >>> clf.fit(X_train, y_train)
        >>> clf.summary()
        ============================================================
          soft-net  |  SoftClassifier
        ============================================================
          Task           : multiclass (3 classes)
          Loss           : sparse_categorical_crossentropy
          Output activ.  : softmax
          ...
        -------- keras model ----------------------------------------
        Model: "sequential"
        ...

        See Also
        --------
        explain : Returns configuration as a string (no Keras summary).
        """
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
        """
        Return a human-readable explanation of the resolved configuration.

        Unlike ``summary()``, this method returns a plain string and does
        not print the Keras architecture. Useful for logging or notebooks.

        Returns
        -------
        explanation : str
            Multi-line string describing the detected task, chosen loss,
            output activation, metrics, and the rationale behind the choice.

        Examples
        --------
        >>> clf.fit(X_train, y_train)
        >>> print(clf.explain())
        Task: multiclass (3 classes)
        Loss: sparse_categorical_crossentropy
        Output activation: softmax
        Metrics: ['accuracy']
        Rationale: 3 unique int labels → multiclass classification

        See Also
        --------
        summary : Prints full Keras architecture on top of the config.
        """
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

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "SoftEstimator":
        """
        Utwórz estymator z predefiniowanej architektury MLP.

        Wczytuje ``layers``, ``dropout`` i ``batch_norm`` z rejestru presetów.
        Wszystkie pozostałe parametry (``epochs``, ``batch_size`` itp.)
        można nadpisać przez ``**kwargs``.

        Parameters
        ----------
        name : str
            Nazwa presetu. Dostępne presety → ``list_presets()``.
            Wbudowane: ``"tiny"``, ``"small"``, ``"medium"``, ``"large"``,
            ``"deep"``, ``"wide"`` i inne.

        **kwargs
            Dowolne parametry ``__init__`` nadpisujące wartości z presetu.
            Np. ``epochs=200``, ``batch_size=64``, ``verbose=1``.

        Returns
        -------
        estimator : instance of the calling class
            Nowa, niefitowana instancja.

        Raises
        ------
        KeyError
            Jeśli preset o podanej nazwie nie istnieje.

        Examples
        --------
        >>> from softnet import SoftClassifier, SoftRegressor
        >>> from softnet.presets import list_presets

        >>> list_presets()   # podejrzyj dostępne presety

        >>> # klasyfikator z presetem "medium"
        >>> clf = SoftClassifier.from_preset("medium", epochs=100, verbose=1)
        >>> clf.fit(X_train, y_train)

        >>> # regresor z presetem "deep", własny batch_size
        >>> reg = SoftRegressor.from_preset("deep", epochs=200, batch_size=64)
        >>> reg.fit(X_train, y_train)

        >>> # własny preset z pliku
        >>> from softnet.presets import load_presets_from_toml
        >>> load_presets_from_toml("~/my_presets.toml")
        >>> clf = SoftClassifier.from_preset("fraud_net", epochs=50)

        See Also
        --------
        softnet.presets.list_presets       : Wypisz dostępne presety.
        softnet.presets.register_preset    : Zarejestruj własny preset.
        softnet.presets.load_presets_from_toml : Wczytaj presety z TOML.
        """
        from softnet.presets import get_preset
        preset = get_preset(name)
        params = dict(
            layers=preset.layers,
            dropout=preset.dropout,
            batch_norm=preset.batch_norm,
        )
        params.update(kwargs)
        return cls(**params)

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
