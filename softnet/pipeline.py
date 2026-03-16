"""
soft-net SoftPipeline — serializowalny pipeline produkcyjny.

Po wytrenowaniu workflow (``wf.run(X, y, ...)``) konwertuj wynik do
``SoftPipeline``, który zawiera tylko transformacje i model potrzebne
do inferencji (bez split, fit, validate). Pipeline można zapisać na dysk
i załadować na serwerze produkcyjnym — bez dostępu do danych treningowych.

Przykłady
---------
Trening (na maszynie deweloperskiej):

>>> from softnet import SoftClassifier
>>> from softnet.workflows import SoftWorkflow
>>> from softnet.pipeline import SoftPipeline
>>>
>>> wf  = SoftWorkflow.from_preset("imb_leaf_robust")
>>> clf = SoftClassifier.from_preset("imb_funnel_heavy", epochs=50)
>>> result = wf.run(X_train, y_train, estimator=clf)
>>>
>>> pipe = result.to_pipeline()
>>> pipe.save("models/fraud_v1")   # zapisuje katalog models/fraud_v1/

Inferencja (serwer produkcyjny — brak y, brak trenowania):

>>> from softnet.pipeline import SoftPipeline
>>> pipe = SoftPipeline.load("models/fraud_v1")
>>>
>>> predictions   = pipe.predict(X_new)          # klasy
>>> probabilities = pipe.predict_proba(X_new)    # prawdopodobieństwa [n, n_classes]
>>> scores        = pipe.decision_score(X_new)   # surowy output modelu

>>> pipe.summary()   # przegląd transformacji i metadanych
"""

from __future__ import annotations

import json
import zipfile
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Wersja formatu zapisu (do weryfikacji przy ładowaniu)
_FORMAT_VERSION = "2"


# ── Reprezentacja jednej transformacji ────────────────────────────────────────

@dataclass
class _Transform:
    """Pojedyncza transformacja w łańcuchu inferencji."""
    name: str          # np. "scaler", "pca", "feature_selector", "leaf_encoding"
    obj: Any           # główny obiekt sklearn (scaler / pca / selector / RF)
    # pola specyficzne dla leaf_encoding (zachowane dla wstecznej kompatybilności)
    ohe: Any = None                       # OneHotEncoder
    orig_idx: list[int] | None = None     # indeksy oryginalnych cech do dołączenia
    # dowolne dodatkowe obiekty/dane dla niestandardowych kroków
    extra: dict = field(default_factory=dict)


# ── Rejestr handlerów transformacji ──────────────────────────────────────────

def _apply_scaler(t: _Transform, X: np.ndarray) -> np.ndarray:
    return t.obj.transform(X)

def _apply_pca(t: _Transform, X: np.ndarray) -> np.ndarray:
    return t.obj.transform(X)

def _apply_feature_selector(t: _Transform, X: np.ndarray) -> np.ndarray:
    return t.obj.transform(X)

def _apply_leaf_encoding(t: _Transform, X: np.ndarray) -> np.ndarray:
    ohe      = t.ohe or t.extra.get("ohe")
    orig_idx = t.orig_idx if t.orig_idx is not None else t.extra.get("orig_idx")
    leaves   = t.obj.apply(X)
    embed    = ohe.transform(leaves)
    if orig_idx is not None:
        return np.hstack([embed, X[:, orig_idx]])
    return embed

# Globalny rejestr: name → fn(transform, X) → X
TRANSFORM_HANDLERS: dict[str, Any] = {
    "scaler":           _apply_scaler,
    "pca":              _apply_pca,
    "feature_selector": _apply_feature_selector,
    "leaf_encoding":    _apply_leaf_encoding,
}


def register_transform_handler(name: str, fn) -> None:
    """
    Zarejestruj handler dla niestandardowego kroku transformacji.

    Pozwala rozszerzać pipeline o własne typy transformacji bez modyfikowania
    kodu soft-net. Handler musi przyjmować ``(_Transform, np.ndarray)``
    i zwracać ``np.ndarray``.

    Parameters
    ----------
    name : str
        Nazwa transformacji (musi być zgodna z ``_Transform.name``).
    fn : callable
        Funkcja ``(transform: _Transform, X: np.ndarray) -> np.ndarray``.

    Examples
    --------
    >>> from softnet.pipeline import register_transform_handler, _Transform
    >>> import numpy as np
    >>>
    >>> def _apply_my_step(t: _Transform, X: np.ndarray) -> np.ndarray:
    ...     num_transformer = t.extra["num_transformer"]
    ...     cat_encoder     = t.extra["cat_encoder"]
    ...     num_idx         = t.extra["num_idx"]
    ...     cat_idx         = t.extra["cat_idx"]
    ...     X_num = num_transformer.transform(X[:, num_idx])
    ...     X_cat = cat_encoder.transform(X[:, cat_idx])
    ...     return np.hstack([X_num, X_cat])
    >>>
    >>> register_transform_handler("column_transform", _apply_my_step)
    """
    TRANSFORM_HANDLERS[name] = fn


# ── SoftPipeline ──────────────────────────────────────────────────────────────

class SoftPipeline:
    """
    Produkcyjny pipeline inferencji — transform + predict bez danych treningowych.

    Tworzony przez ``WorkflowResult.to_pipeline()`` po wytrenowaniu workflow.
    Zawiera wszystkie dopasowane transformacje (scaler, PCA, feature selector,
    leaf encoding) oraz wytrenowany model.

    Na produkcji nie potrzebujesz ``y`` ani żadnego kroku treningowego —
    tylko ``predict(X_new)``.

    Parameters
    ----------
    transforms : list of _Transform
        Łańcuch transformacji w kolejności zastosowania.
    estimator : sklearn-compatible estimator
        Wytrenowany model (SoftClassifier / SoftRegressor / VotingClassifier itp.)
    metadata : dict
        Metadane: wersja, task_type, metryki z treningu itp.

    Examples
    --------
    Zapis i odczyt:

    >>> pipe = result.to_pipeline()
    >>> pipe.save("models/fraud_v1")
    >>> # --- na produkcji ---
    >>> pipe = SoftPipeline.load("models/fraud_v1")
    >>> pipe.predict(X_new)

    Jako zip (jeden plik do wdrożenia):

    >>> pipe.save("models/fraud_v1", as_zip=True)   # → fraud_v1.softpipe
    >>> pipe = SoftPipeline.load("models/fraud_v1.softpipe")

    See Also
    --------
    WorkflowResult.to_pipeline : Fabryka pipeline z wyniku workflow.
    """

    def __init__(
        self,
        transforms: list[_Transform],
        estimator: Any,
        metadata: dict | None = None,
    ):
        self.transforms = transforms
        self.estimator = estimator
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Inferencja
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Przeprowadź X przez wszystkie transformacje (bez modelu)."""
        for t in self.transforms:
            handler = TRANSFORM_HANDLERS.get(t.name)
            if handler is None:
                raise ValueError(
                    f"Nieznany typ transformacji: {t.name!r}. "
                    f"Zarejestruj handler przez: "
                    f"softnet.pipeline.register_transform_handler('{t.name}', fn)"
                )
            X = handler(t, X)
        return X

    def predict(self, X) -> np.ndarray:
        """
        Przewiduj klasy / wartości dla nowych danych.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Surowe dane — takie same cechy jak podczas treningu (przed skalowaniem).

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Przewidywane klasy (klasyfikacja) lub wartości (regresja).

        Examples
        --------
        >>> pipe = SoftPipeline.load("models/fraud_v1")
        >>> y_pred = pipe.predict(X_new)
        """
        from sklearn.utils.validation import check_array
        X = check_array(X)
        X = self._transform(X)
        return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Zwróć prawdopodobieństwa klas (tylko klasyfikacja).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Surowe dane wejściowe.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Prawdopodobieństwa. Dla binarnej klasyfikacji: kolumna 1 to P(fraud).

        Raises
        ------
        AttributeError
            Jeśli model nie obsługuje ``predict_proba`` (np. regressor).

        Examples
        --------
        >>> proba = pipe.predict_proba(X_new)
        >>> fraud_score = proba[:, 1]   # prawdopodobieństwo klasy pozytywnej
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                "Ten model nie obsługuje predict_proba. "
                "Użyj predict() dla regresorów lub modeli bez prawdopodobieństw."
            )
        from sklearn.utils.validation import check_array
        X = check_array(X)
        X = self._transform(X)
        return self.estimator.predict_proba(X)

    def decision_score(self, X) -> np.ndarray:
        """
        Surowy output sieci neuronowej przed progowaniem.

        Przydatne gdy chcesz sam dobrać próg decyzji (np. dla imbalanced data).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Surowe dane wejściowe.

        Returns
        -------
        scores : ndarray
            Dla SoftClassifier/SoftRegressor: surowy output Keras (predict).
            Dla innych estimatorów: deleguje do predict_proba lub predict.

        Examples
        --------
        >>> scores = pipe.decision_score(X_new)
        >>> custom_threshold = 0.3   # zamiast domyślnych 0.5
        >>> y_pred = (scores[:, 0] >= custom_threshold).astype(int)
        """
        from sklearn.utils.validation import check_array
        X = check_array(X)
        X = self._transform(X)

        # SoftClassifier / SoftRegressor mają model_ (Keras)
        if hasattr(self.estimator, "model_") and self.estimator.model_ is not None:
            return self.estimator.model_.predict(X)

        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)

        return self.estimator.predict(X)

    # ------------------------------------------------------------------
    # Zapis i odczyt
    # ------------------------------------------------------------------

    def save(self, path: str | Path, as_zip: bool = False) -> Path:
        """
        Zapisz pipeline na dysk.

        Domyślnie zapisuje jako katalog. Opcjonalnie jako plik ``.softpipe``
        (zip) — wygodny do wdrożenia jako jeden plik.

        Parameters
        ----------
        path : str or Path
            Ścieżka katalogu (lub pliku ``.softpipe`` gdy ``as_zip=True``).
            Rozszerzenie ``.softpipe`` dodawane automatycznie gdy ``as_zip=True``.
        as_zip : bool, default False
            Czy zapisać jako zip (``.softpipe``).

        Returns
        -------
        saved_path : Path
            Faktyczna ścieżka zapisu.

        Examples
        --------
        >>> pipe.save("models/fraud_v1")              # → katalog
        >>> pipe.save("models/fraud_v1", as_zip=True) # → models/fraud_v1.softpipe
        """
        import joblib

        path = Path(path)

        if as_zip:
            # Zapisz najpierw do tymczasowego katalogu, potem spakuj
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp) / "pipeline"
                self._save_to_dir(tmp_dir)
                zip_path = path.with_suffix(".softpipe")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in tmp_dir.rglob("*"):
                        zf.write(f, f.relative_to(tmp_dir))
            print(f"[soft-net] Pipeline zapisany → {zip_path}")
            return zip_path
        else:
            self._save_to_dir(path)
            print(f"[soft-net] Pipeline zapisany → {path}/")
            return path

    def _save_to_dir(self, directory: Path) -> None:
        """Zapisz pipeline do katalogu."""
        import joblib

        directory.mkdir(parents=True, exist_ok=True)

        # 1. metadata
        meta = dict(self.metadata)
        meta["format_version"] = _FORMAT_VERSION
        meta["transforms"] = []

        # 2. transformacje
        _PRIMITIVES = (str, int, float, bool, type(None), list, dict)
        for i, t in enumerate(self.transforms):
            t_meta: dict[str, Any] = {"name": t.name, "index": i}

            if t.name == "leaf_encoding":
                joblib.dump(t.obj, directory / f"transform_{i}_rf.joblib")
                joblib.dump(t.ohe, directory / f"transform_{i}_ohe.joblib")
                t_meta["orig_idx"] = t.orig_idx
            else:
                joblib.dump(t.obj, directory / f"transform_{i}.joblib")

            # extra — obiekty sklearn jako joblib, prymitywy w JSON
            if t.extra:
                extra_keys_obj = []
                extra_primitives = {}
                for key, val in t.extra.items():
                    if isinstance(val, _PRIMITIVES):
                        extra_primitives[key] = val
                    else:
                        joblib.dump(val, directory / f"transform_{i}_extra_{key}.joblib")
                        extra_keys_obj.append(key)
                t_meta["extra_obj_keys"] = extra_keys_obj
                t_meta["extra_primitives"] = extra_primitives

            meta["transforms"].append(t_meta)

        # 3. model
        if hasattr(self.estimator, "model_") and self.estimator.model_ is not None:
            # SoftClassifier / SoftRegressor — zapisz Keras model
            self.estimator.model_.save(directory / "model.keras")
            meta["model_format"] = "keras_estimator"
            joblib.dump(self.estimator, directory / "estimator.joblib")
        else:
            # sklearn / voting / inne
            joblib.dump(self.estimator, directory / "estimator.joblib")
            meta["model_format"] = "joblib"

        (directory / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SoftPipeline":
        """
        Załaduj pipeline z dysku.

        Obsługuje zarówno katalog jak i plik ``.softpipe``.

        Parameters
        ----------
        path : str or Path
            Ścieżka do katalogu lub pliku ``.softpipe``.

        Returns
        -------
        SoftPipeline

        Examples
        --------
        >>> pipe = SoftPipeline.load("models/fraud_v1")
        >>> pipe = SoftPipeline.load("models/fraud_v1.softpipe")
        >>> y_pred = pipe.predict(X_new)
        """
        import joblib

        path = Path(path)

        # Jeśli zip — rozpakuj do tymczasowego katalogu
        if path.suffix == ".softpipe" or zipfile.is_zipfile(path):
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmp)
                return cls._load_from_dir(Path(tmp))
        else:
            return cls._load_from_dir(path)

    @classmethod
    def _load_from_dir(cls, directory: Path) -> "SoftPipeline":
        """Załaduj pipeline z katalogu."""
        import joblib

        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        version = meta.get("format_version", "unknown")
        if version != _FORMAT_VERSION:
            import warnings
            warnings.warn(
                f"Format pipeline v{version} różni się od bieżącego v{_FORMAT_VERSION}. "
                "Możliwe problemy ze zgodnością.",
                UserWarning,
                stacklevel=2,
            )

        # Transformacje
        transforms = []
        for t_meta in meta.get("transforms", []):
            i = t_meta["index"]
            name = t_meta["name"]

            if name == "leaf_encoding":
                rf  = joblib.load(directory / f"transform_{i}_rf.joblib")
                ohe = joblib.load(directory / f"transform_{i}_ohe.joblib")
                orig_idx = t_meta.get("orig_idx")
                transforms.append(_Transform(name=name, obj=rf, ohe=ohe, orig_idx=orig_idx))
            else:
                obj = joblib.load(directory / f"transform_{i}.joblib")

                # extra — odtwórz obiekty i prymitywy
                extra: dict = {}
                for key in t_meta.get("extra_obj_keys", []):
                    extra[key] = joblib.load(directory / f"transform_{i}_extra_{key}.joblib")
                extra.update(t_meta.get("extra_primitives", {}))

                transforms.append(_Transform(name=name, obj=obj, extra=extra))

        # Model
        estimator = joblib.load(directory / "estimator.joblib")

        # Jeśli Keras model był zapisany osobno — załaduj i podepnij
        keras_path = directory / "model.keras"
        if keras_path.exists() and meta.get("model_format") == "keras_estimator":
            try:
                import keras
                estimator.model_ = keras.saving.load_model(keras_path)
            except Exception:
                pass  # estimator.joblib może już mieć model_

        return cls(transforms=transforms, estimator=estimator, metadata=meta)

    # ------------------------------------------------------------------
    # Podgląd
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """
        Wypisz podsumowanie pipeline'u.

        Examples
        --------
        >>> pipe.summary()
        ╔══════════════════════════════════════╗
        ║  SoftPipeline — production pipeline  ║
        ╠══════════════════════════════════════╣
        ║  Transformacje (2):                  ║
        ║    1. scaler   → RobustScaler        ║
        ║    2. leaf_enc → RF(200) + OHE       ║
        ║  Model: SoftClassifier (Keras)       ║
        ║  Metryki treningu:                   ║
        ║    accuracy: 0.9991                  ║
        ╚══════════════════════════════════════╝
        """
        print("╔══════════════════════════════════════════╗")
        print("║  SoftPipeline — production pipeline      ║")
        print("╠══════════════════════════════════════════╣")

        print(f"║  Transformacje ({len(self.transforms)}):{'':25}║")
        for i, t in enumerate(self.transforms, 1):
            if t.name == "leaf_encoding":
                n_trees = getattr(t.obj, "n_estimators", "?")
                detail = f"RF({n_trees}) + OHE"
            elif t.name == "scaler":
                detail = type(t.obj).__name__
            elif t.name == "pca":
                nc = getattr(t.obj, "n_components_", getattr(t.obj, "n_components", "?"))
                detail = f"PCA(n={nc})"
            elif t.name == "feature_selector":
                detail = f"SelectFromModel"
            else:
                detail = type(t.obj).__name__
            line = f"    {i}. {t.name:<20} → {detail}"
            print(f"║  {line:<40}║")

        est_name = type(self.estimator).__name__
        has_keras = hasattr(self.estimator, "model_") and self.estimator.model_ is not None
        keras_tag = " (Keras)" if has_keras else ""
        print(f"║  Model: {est_name}{keras_tag:<30}║")

        # Metryki z treningu (jeśli są w metadata)
        metrics = {
            k: v for k, v in self.metadata.items()
            if k in ("accuracy", "f1", "roc_auc", "r2", "mae", "rmse", "score")
        }
        if metrics:
            print(f"║  Metryki treningu:{'':22}║")
            for k, v in metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"║    {k}: {val_str:<34}║")

        print("╚══════════════════════════════════════════╝")

    def __repr__(self) -> str:
        transform_names = [t.name for t in self.transforms]
        return (
            f"SoftPipeline("
            f"transforms={transform_names}, "
            f"estimator={type(self.estimator).__name__})"
        )
