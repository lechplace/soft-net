"""
soft-net workflow steps.

Każdy krok operuje na wspólnym słowniku kontekstu (ctx) i go zwraca.
ctx zawiera dane, model, wyniki — przekazywane między krokami.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── kontekst ─────────────────────────────────────────────────────────────────

Context = dict[str, Any]


# ── SplitStep ─────────────────────────────────────────────────────────────────

class SplitStep:
    """
    Podział danych na zbiór treningowy i testowy.

    Parameters
    ----------
    test_size : float, default 0.2
        Ułamek danych przeznaczony na zbiór testowy.
    stratify : bool, default True
        Czy użyć stratyfikacji (tylko dla klasyfikacji).
        Automatycznie wyłączana dla regresji.
    random_state : int, default 42
        Seed podziału.

    Examples
    --------
    >>> step = SplitStep(test_size=0.25, random_state=0)
    """

    def __init__(self, test_size: float = 0.2, stratify: bool = True, random_state: int = 42):
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

    def run(self, ctx: Context) -> Context:
        from sklearn.model_selection import train_test_split

        X, y = ctx["X"], ctx["y"]
        # stratify tylko gdy nie-regresja i liczba unikalnych klas < 50% próbek
        use_stratify = (
            self.stratify
            and np.issubdtype(y.dtype, np.integer)
            and len(np.unique(y)) < len(y) * 0.5
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y if use_stratify else None,
            random_state=self.random_state,
        )
        ctx.update({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        })
        return ctx


# ── ScaleStep ─────────────────────────────────────────────────────────────────

class ScaleStep:
    """
    Normalizacja cech (fit na train, transform na test i train).

    Parameters
    ----------
    method : {"standard", "minmax", "robust"}, default "standard"
        Metoda skalowania.
        - ``"standard"`` — zero mean, unit variance (StandardScaler)
        - ``"minmax"``   — zakres [0, 1] (MinMaxScaler)
        - ``"robust"``   — odporny na outliery (RobustScaler)

    Examples
    --------
    >>> step = ScaleStep(method="robust")
    """

    SCALERS = {
        "standard": "sklearn.preprocessing.StandardScaler",
        "minmax":   "sklearn.preprocessing.MinMaxScaler",
        "robust":   "sklearn.preprocessing.RobustScaler",
    }

    def __init__(self, method: str = "standard"):
        if method not in self.SCALERS:
            raise ValueError(f"method musi być jednym z: {list(self.SCALERS)}")
        self.method = method

    def run(self, ctx: Context) -> Context:
        import importlib
        module_path, cls_name = self.SCALERS[self.method].rsplit(".", 1)
        ScalerCls = getattr(importlib.import_module(module_path), cls_name)

        scaler = ScalerCls()
        ctx["X_train"] = scaler.fit_transform(ctx["X_train"])
        ctx["X_test"]  = scaler.transform(ctx["X_test"])
        ctx["scaler"]  = scaler
        return ctx


# ── FitStep ───────────────────────────────────────────────────────────────────

class FitStep:
    """
    Trenowanie estymatora na danych treningowych.

    Estimator musi być przekazany przez ``SoftWorkflow.run(estimator=...)``.

    Examples
    --------
    >>> step = FitStep()
    """

    def run(self, ctx: Context) -> Context:
        estimator = ctx.get("estimator")
        if estimator is None:
            raise ValueError(
                "FitStep wymaga estymatora. "
                "Przekaż go przez workflow.run(X, y, estimator=...)."
            )
        estimator.fit(ctx["X_train"], ctx["y_train"])
        ctx["fitted_estimator"] = estimator
        return ctx


# ── ValidateStep ──────────────────────────────────────────────────────────────

class ValidateStep:
    """
    Ewaluacja modelu na zbiorze testowym.

    Metryki dobierane automatycznie:
    - klasyfikacja binarna     → accuracy, F1, AUC-ROC, classification_report
    - klasyfikacja wieloklasowa → accuracy, F1-macro, classification_report
    - regresja                 → R², MAE, RMSE

    Examples
    --------
    >>> step = ValidateStep()
    """

    def run(self, ctx: Context) -> Context:
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report,
            roc_auc_score, r2_score, mean_absolute_error,
            mean_squared_error,
        )
        from softnet.inference import TaskInferrer, TaskType

        estimator = ctx.get("fitted_estimator") or ctx.get("estimator")
        X_test, y_test = ctx["X_test"], ctx["y_test"]
        y_pred = estimator.predict(X_test)

        task = TaskInferrer().infer(ctx["y_train"])
        results: dict[str, Any] = {}

        if task.task_type == TaskType.REGRESSION:
            results["r2"]   = r2_score(y_test, y_pred)
            results["mae"]  = mean_absolute_error(y_test, y_pred)
            results["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results["score"] = results["r2"]
            results["report"] = (
                f"R²={results['r2']:.4f}  "
                f"MAE={results['mae']:.4f}  "
                f"RMSE={results['rmse']:.4f}"
            )
        else:
            results["accuracy"] = accuracy_score(y_test, y_pred)
            avg = "binary" if task.task_type == TaskType.BINARY else "macro"
            results["f1"] = f1_score(y_test, y_pred, average=avg, zero_division=0)
            results["classification_report"] = classification_report(y_test, y_pred)
            results["score"] = results["accuracy"]
            results["report"] = results["classification_report"]

            if task.task_type == TaskType.BINARY and hasattr(estimator, "predict_proba"):
                try:
                    proba = estimator.predict_proba(X_test)[:, 1]
                    results["roc_auc"] = roc_auc_score(y_test, proba)
                except Exception:
                    pass

        ctx["validation"] = results
        return ctx


# ── SaveStep ──────────────────────────────────────────────────────────────────

class SaveStep:
    """
    Zapis wytrenowanego modelu na dysk.

    Parameters
    ----------
    path : str, default "model"
        Ścieżka zapisu bez rozszerzenia.
        Rozszerzenie dodawane automatycznie (``.keras`` lub ``.joblib``).
    format : {"keras", "joblib"}, default "keras"
        Format zapisu.
        - ``"keras"``  — natywny format Keras (tylko soft-net estimatory)
        - ``"joblib"`` — pickle-compatible (dowolny sklearn estimator)

    Examples
    --------
    >>> step = SaveStep(path="models/fraud_detector", format="keras")
    """

    def __init__(self, path: str = "model", format: str = "keras"):
        self.path = path
        self.format = format

    def run(self, ctx: Context) -> Context:
        estimator = ctx.get("fitted_estimator") or ctx.get("estimator")

        if self.format == "keras":
            save_path = f"{self.path}.keras"
            if hasattr(estimator, "model_") and estimator.model_ is not None:
                estimator.model_.save(save_path)
            else:
                raise ValueError("SaveStep(format='keras'): model_ nie istnieje. Użyj format='joblib'.")
        elif self.format == "joblib":
            import joblib
            save_path = f"{self.path}.joblib"
            joblib.dump(estimator, save_path)
        else:
            raise ValueError(f"format musi być 'keras' lub 'joblib', nie '{self.format}'")

        ctx["saved_path"] = save_path
        print(f"[soft-net] Model zapisany → {save_path}")
        return ctx


# ── GridSearchStep ────────────────────────────────────────────────────────────

class GridSearchStep:
    """
    Przeszukiwanie siatki hiperparametrów (GridSearchCV).

    Parameters
    ----------
    param_grid : dict, optional
        Siatka parametrów. Domyślnie przeszukuje ``epochs`` i ``dropout``
        dla soft-net estimatorów.
    cv : int, default 5
        Liczba foldów cross-validation.
    scoring : str, optional
        Metryka scoringu. Dobierana automatycznie jeśli nie podana.
    n_jobs : int, default -1
        Liczba równoległych procesów. ``-1`` = wszystkie rdzenie.
    verbose : int, default 0
        Poziom szczegółowości logów GridSearchCV.

    Examples
    --------
    >>> step = GridSearchStep(
    ...     param_grid={"epochs": [50, 100], "dropout": [0.1, 0.3]},
    ...     cv=3,
    ... )
    """

    DEFAULT_PARAM_GRID = {
        "epochs":  [50, 100],
        "dropout": [0.1, 0.2, 0.3],
    }

    def __init__(
        self,
        param_grid: dict | None = None,
        cv: int = 5,
        scoring: str | None = None,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self, ctx: Context) -> Context:
        from sklearn.model_selection import GridSearchCV

        estimator = ctx.get("estimator")
        if estimator is None:
            raise ValueError("GridSearchStep wymaga estymatora w ctx['estimator'].")

        gs = GridSearchCV(
            estimator,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True,
        )
        gs.fit(ctx["X_train"], ctx["y_train"])

        ctx["grid_search"]     = gs
        ctx["best_params"]     = gs.best_params_
        ctx["best_score_cv"]   = gs.best_score_
        ctx["fitted_estimator"] = gs.best_estimator_

        print(f"[soft-net] GridSearch najlepsze params: {gs.best_params_}  "
              f"CV score: {gs.best_score_:.4f}")
        return ctx


# ── RandomizedSearchStep ──────────────────────────────────────────────────────

class RandomizedSearchStep:
    """
    Losowe przeszukiwanie hiperparametrów (RandomizedSearchCV).

    Szybsza alternatywa dla ``GridSearchStep`` — przeszukuje losowy podzbiór
    siatki parametrów.

    Parameters
    ----------
    param_distributions : dict, optional
        Rozkłady parametrów. Domyślnie: ``epochs`` i ``dropout``.
    n_iter : int, default 20
        Liczba losowanych kombinacji.
    cv : int, default 5
        Liczba foldów cross-validation.
    scoring : str, optional
        Metryka scoringu. Dobierana automatycznie jeśli nie podana.
    n_jobs : int, default -1
        Liczba równoległych procesów.
    random_state : int, default 42
        Seed losowania.

    Examples
    --------
    >>> from scipy.stats import randint, uniform
    >>> step = RandomizedSearchStep(
    ...     param_distributions={"epochs": randint(50, 200), "dropout": uniform(0, 0.5)},
    ...     n_iter=10,
    ... )
    """

    DEFAULT_PARAM_DISTRIBUTIONS = {
        "epochs":  [50, 75, 100, 150, 200],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    }

    def __init__(
        self,
        param_distributions: dict | None = None,
        n_iter: int = 20,
        cv: int = 5,
        scoring: str | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.param_distributions = param_distributions or self.DEFAULT_PARAM_DISTRIBUTIONS
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def run(self, ctx: Context) -> Context:
        from sklearn.model_selection import RandomizedSearchCV

        estimator = ctx.get("estimator")
        if estimator is None:
            raise ValueError("RandomizedSearchStep wymaga estymatora w ctx['estimator'].")

        rs = RandomizedSearchCV(
            estimator,
            self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            refit=True,
        )
        rs.fit(ctx["X_train"], ctx["y_train"])

        ctx["random_search"]   = rs
        ctx["best_params"]     = rs.best_params_
        ctx["best_score_cv"]   = rs.best_score_
        ctx["fitted_estimator"] = rs.best_estimator_

        print(f"[soft-net] RandomizedSearch najlepsze params: {rs.best_params_}  "
              f"CV score: {rs.best_score_:.4f}")
        return ctx


# ── VotingStep ────────────────────────────────────────────────────────────────

class VotingStep:
    """
    Ansambl głosujący (VotingClassifier / VotingRegressor).

    Domyślnie tworzy trzech głosujących: soft-net ``small``, ``medium``,
    ``large``. Możesz podmienić listę estimatorów przez ``estimators``.

    Parameters
    ----------
    estimators : list of (name, estimator) tuples, optional
        Głosujące estymatory. Domyślnie trzy SoftClassifier z presetów.
    voting : {"soft", "hard"}, default "soft"
        Tryb głosowania.
        - ``"soft"`` — uśrednienie prawdopodobieństw (wymaga predict_proba)
        - ``"hard"`` — głosowanie większościowe

    Examples
    --------
    >>> from softnet import SoftClassifier
    >>> step = VotingStep(
    ...     estimators=[
    ...         ("small",  SoftClassifier.from_preset("small",  epochs=50)),
    ...         ("medium", SoftClassifier.from_preset("medium", epochs=50)),
    ...     ],
    ...     voting="soft",
    ... )
    """

    def __init__(self, estimators: list | None = None, voting: str = "soft"):
        self.estimators = estimators
        self.voting = voting

    def _default_estimators(self, task_type):
        from softnet.tabular import SoftClassifier, SoftRegressor

        if str(task_type).startswith("TaskType.REGRESSION") or "regression" in str(task_type).lower():
            EstCls = SoftRegressor
        else:
            EstCls = SoftClassifier

        return [
            ("small",  EstCls.from_preset("small",  epochs=50)),
            ("medium", EstCls.from_preset("medium", epochs=50)),
            ("large",  EstCls.from_preset("large",  epochs=50)),
        ]

    def run(self, ctx: Context) -> Context:
        from softnet.inference import TaskInferrer, TaskType

        task = TaskInferrer().infer(ctx["y_train"])
        estimators = self.estimators or self._default_estimators(task.task_type)

        if task.task_type == TaskType.REGRESSION:
            from sklearn.ensemble import VotingRegressor
            ensemble = VotingRegressor(estimators=estimators)
        else:
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(estimators=estimators, voting=self.voting)

        ensemble.fit(ctx["X_train"], ctx["y_train"])
        ctx["fitted_estimator"] = ensemble
        ctx["voting_ensemble"]  = ensemble
        return ctx


# ── rejestr kroków ────────────────────────────────────────────────────────────

STEP_REGISTRY: dict[str, type] = {
    "split":             SplitStep,
    "scale":             ScaleStep,
    "fit":               FitStep,
    "validate":          ValidateStep,
    "save":              SaveStep,
    "grid_search":       GridSearchStep,
    "randomized_search": RandomizedSearchStep,
    "voting":            VotingStep,
}
