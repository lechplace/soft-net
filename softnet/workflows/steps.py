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
        ctx.setdefault("_transform_chain", []).append({"name": "scaler", "obj": scaler})
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

        if task.task == TaskType.REGRESSION:
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
            avg = "binary" if task.task == TaskType.BINARY else "macro"
            results["f1"] = f1_score(y_test, y_pred, average=avg, zero_division=0)
            results["classification_report"] = classification_report(y_test, y_pred)
            results["score"] = results["accuracy"]
            results["report"] = results["classification_report"]

            if task.task == TaskType.BINARY and hasattr(estimator, "predict_proba"):
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
        from softnet.inference import TaskType

        if task_type == TaskType.REGRESSION or task_type == TaskType.MULTIOUTPUT_REGRESSION:
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
        estimators = self.estimators or self._default_estimators(task.task)

        if task.task == TaskType.REGRESSION:
            from sklearn.ensemble import VotingRegressor
            ensemble = VotingRegressor(estimators=estimators)
        else:
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(estimators=estimators, voting=self.voting)

        ensemble.fit(ctx["X_train"], ctx["y_train"])
        ctx["fitted_estimator"] = ensemble
        ctx["voting_ensemble"]  = ensemble
        return ctx


# ── PCAStep ───────────────────────────────────────────────────────────────────

class PCAStep:
    """
    Redukcja wymiarowości przez PCA (Principal Component Analysis).

    Musi być użyty **po** ``SplitStep`` — PCA jest fitowany wyłącznie
    na danych treningowych, a następnie transformuje zarówno train jak i test.
    Dzięki temu nie dochodzi do data leakage.

    Parameters
    ----------
    n_components : int, float or "mle", optional
        Liczba komponentów głównych:

        - ``int``   — dokładna liczba składowych, np. ``50``
        - ``float`` — ułamek wyjaśnianej wariancji, np. ``0.95`` (95%)
        - ``"mle"`` — automatyczny dobór przez MLE (tylko dla n_samples > n_features)
        - ``None``  — wszystkie komponenty (brak redukcji, tylko rotacja)

        Domyślnie ``0.95`` — zachowaj 95% wariancji.
    whiten : bool, default False
        Czy wybielić komponenty (normalizacja wariancji każdego PC do 1).
        Przydatne przed sieciami neuronowymi gdy cechy mają różne skale.
    random_state : int, default 42
        Seed dla algorytmu randomized SVD.

    Attributes set in ctx
    ---------------------
    ctx["pca"] : sklearn.decomposition.PCA
        Fitowany obiekt PCA. Dostępny po wykonaniu kroku przez
        ``result.ctx["pca"]``.
    ctx["pca_explained_variance"] : float
        Łączny odsetek wyjaśnionej wariancji (suma ``explained_variance_ratio_``).
    ctx["pca_n_components"] : int
        Faktyczna liczba użytych komponentów.

    Notes
    -----
    Kolejność kroków z PCA:

    ``split → scale → pca → fit → validate``

    PCA po skalowaniu daje lepsze wyniki — StandardScaler zapewnia
    równe wagi cech przed obliczaniem wariancji.

    Examples
    --------
    Zachowaj 95% wariancji (domyślne):

    >>> step = PCAStep()

    Dokładna liczba komponentów:

    >>> step = PCAStep(n_components=20)

    W workflow:

    >>> from softnet.workflows import SoftWorkflow, PCAStep
    >>> wf = SoftWorkflow.from_preset(
    ...     "scaled",
    ...     step_overrides={"pca": PCAStep(n_components=0.99)},
    ... )

    Lub przez preset ``scaled_pca``:

    >>> wf = SoftWorkflow.from_preset("scaled_pca")
    >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
    >>> print(result.ctx["pca_n_components"])   # ile składowych zostało użytych
    >>> print(result.ctx["pca_explained_variance"])  # ile % wariancji zachowane
    """

    def __init__(
        self,
        n_components: int | float | str | None = 0.95,
        whiten: bool = False,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def run(self, ctx: Context) -> Context:
        from sklearn.decomposition import PCA

        if "X_train" not in ctx:
            raise ValueError(
                "PCAStep wymaga wcześniejszego SplitStep. "
                "Upewnij się, że 'split' jest przed 'pca' w sekwencji kroków."
            )

        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        ctx["X_train"] = pca.fit_transform(ctx["X_train"])
        ctx["X_test"]  = pca.transform(ctx["X_test"])
        ctx["pca"]     = pca
        ctx["pca_n_components"]       = pca.n_components_
        ctx["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        ctx.setdefault("_transform_chain", []).append({"name": "pca", "obj": pca})

        print(
            f"[soft-net] PCA: {ctx['X_train'].shape[1]} → {pca.n_components_} komponentów  "
            f"({ctx['pca_explained_variance'] * 100:.1f}% wariancji zachowane)"
        )
        return ctx


# ── FeatureSelectionStep ──────────────────────────────────────────────────────

class FeatureSelectionStep:
    """
    Selekcja cech na podstawie feature importance z modelu drzewiastego.

    Uruchamia RandomForest (lub inny model z ``feature_importances_``) na danych
    treningowych, a następnie wybiera najważniejsze cechy i przekazuje je
    do kolejnych kroków (np. ``FitStep``).

    Fitowany wyłącznie na ``X_train`` — brak data leakage.

    Parameters
    ----------
    selector_estimator : sklearn estimator, optional
        Model używany do obliczenia ważności cech.
        Musi posiadać atrybut ``feature_importances_`` po fitowaniu.
        Domyślnie: ``RandomForestClassifier(n_estimators=100, random_state=42)``
        dla klasyfikacji lub ``RandomForestRegressor`` dla regresji.
        Wykrywanie automatyczne na podstawie ``y_train``.
    threshold : str or float, default "mean"
        Próg ważności cechy:

        - ``"mean"``   — cechy powyżej średniej ważności (domyślne)
        - ``"median"`` — cechy powyżej mediany ważności
        - ``float``    — np. ``0.01`` — cechy z ważnością > 1%
    max_features : int or None, default None
        Maksymalna liczba wybieranych cech.
        Jeśli podane, wybierane są top-N według ważności
        (zamiast progu ``threshold``).
    n_estimators : int, default 100
        Liczba drzew w domyślnym RandomForest (ignorowane gdy podano
        własny ``selector_estimator``).
    random_state : int, default 42
        Seed dla domyślnego RandomForest.

    Attributes set in ctx
    ---------------------
    ctx["feature_selector"] : SelectFromModel
        Fitowany selektor — można go użyć do transformacji nowych danych.
    ctx["feature_importances"] : ndarray of shape (n_features,)
        Wektor ważności cech z modelu selekcyjnego (przed filtrowaniem).
    ctx["selected_features"] : list of int
        Indeksy wybranych cech w oryginalnej macierzy X.
    ctx["n_features_original"] : int
        Liczba cech przed selekcją.
    ctx["n_features_selected"] : int
        Liczba cech po selekcji.

    Notes
    -----
    Zalecana kolejność kroków:

    ``split → scale → feature_selection → fit → validate``

    lub z PCA po selekcji (rzadziej potrzebne, ale możliwe):

    ``split → scale → feature_selection → pca → fit → validate``

    Examples
    --------
    Domyślnie — RandomForest, próg = mean:

    >>> step = FeatureSelectionStep()

    Top 20 najważniejszych cech:

    >>> step = FeatureSelectionStep(max_features=20)

    Własny próg:

    >>> step = FeatureSelectionStep(threshold=0.01)

    Własny model selekcji (np. GradientBoosting):

    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> step = FeatureSelectionStep(
    ...     selector_estimator=GradientBoostingClassifier(n_estimators=50),
    ... )

    Przez preset:

    >>> wf = SoftWorkflow.from_preset("rf_select")
    >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
    >>> print(result.ctx["selected_features"])   # indeksy wybranych cech
    >>> print(result.ctx["feature_importances"]) # ważność wszystkich cech
    """

    def __init__(
        self,
        selector_estimator=None,
        threshold: str | float = "mean",
        max_features: int | None = None,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.selector_estimator = selector_estimator
        self.threshold = threshold
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _default_estimator(self, y):
        from softnet.inference import TaskInferrer, TaskType
        task = TaskInferrer().infer(y)
        if task.task == TaskType.REGRESSION:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def run(self, ctx: Context) -> Context:
        import numpy as np
        from sklearn.feature_selection import SelectFromModel

        if "X_train" not in ctx:
            raise ValueError(
                "FeatureSelectionStep wymaga wcześniejszego SplitStep. "
                "Upewnij się, że 'split' jest przed 'feature_selection'."
            )

        X_train, y_train = ctx["X_train"], ctx["y_train"]
        n_original = X_train.shape[1]

        estimator = self.selector_estimator or self._default_estimator(y_train)

        # SelectFromModel: fit na train, transform na train i test
        if self.max_features is not None:
            # tryb top-N: próg = (n+1)-ta najwyższa ważność
            estimator.fit(X_train, y_train)
            importances = estimator.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            cutoff = importances[sorted_idx[min(self.max_features, n_original) - 1]]
            selector = SelectFromModel(estimator, threshold=cutoff, prefit=True)
        else:
            selector = SelectFromModel(estimator, threshold=self.threshold)
            selector.fit(X_train, y_train)
            importances = selector.estimator_.feature_importances_

        X_train_new = selector.transform(X_train)
        X_test_new  = selector.transform(ctx["X_test"])
        selected    = list(selector.get_support(indices=True))

        ctx["X_train"] = X_train_new
        ctx["X_test"]  = X_test_new
        ctx["feature_selector"]    = selector
        ctx["feature_importances"] = importances
        ctx["selected_features"]   = selected
        ctx["n_features_original"] = n_original
        ctx["n_features_selected"] = X_train_new.shape[1]
        ctx.setdefault("_transform_chain", []).append({"name": "feature_selector", "obj": selector})

        est_name = type(selector.estimator_).__name__
        print(
            f"[soft-net] FeatureSelection ({est_name}): "
            f"{n_original} → {X_train_new.shape[1]} cech  "
            f"(próg={self.threshold if self.max_features is None else f'top-{self.max_features}'})"
        )
        return ctx


# ── LeafEncodingStep ──────────────────────────────────────────────────────────

class LeafEncodingStep:
    """
    Tree Embedding: ID liści RandomForest jako nowe cechy dla sieci neuronowej.

    Technika spopularyzowana przez Facebook (Practical Lessons from Predicting
    Clicks on Ads at Facebook, 2014). RandomForest robi dwie rzeczy jednocześnie:

    1. **Embedding** — każda próbka jest przepuszczana przez wszystkie drzewa,
       ID liścia w którym wyląduje staje się cechą binarną (one-hot).
       Wynik: macierz (n_samples, suma_liści_wszystkich_drzew).

    2. **Selekcja** — opcjonalnie dołącza oryginalne cechy o wysokiej ważności
       obok embeddingów.

    Schemat przepływu danych::

        X_train  ──► RF.fit(X_train, y_train)
                         │
                         ├─► RF.apply(X_train)  → liście (n, n_trees)
                         │        └─► OneHotEncode → embedding (n, Σliście)
                         │
                         └─► feature_importances_ → top-K oryginalne cechy
                                                         │
                         embedding ──── concat ──────────┘
                                          │
                                     X_train_new  ──► FitStep (soft-net)

    Parameters
    ----------
    n_estimators : int, default 100
        Liczba drzew w RandomForest.
    max_leaf_nodes : int or None, default 32
        Maksymalna liczba liści na drzewo. Ogranicza wymiarowość embeddingu.
        ``None`` — bez ograniczeń (może dać bardzo dużo cech).
    include_original : bool, default True
        Czy dołączyć oryginalne cechy obok embeddingów z liści.
        Jeśli ``True``, łączy top-``max_original_features`` cech z embeddingiem.
    max_original_features : int or None, default None
        Liczba oryginalnych cech do dołączenia (sortowane według ważności RF).
        ``None`` — wszystkie oryginalne cechy gdy ``include_original=True``.
    importance_threshold : float or None, default None
        Alternatywa dla ``max_original_features`` — dołącz cechy z ważnością
        powyżej progu. Ignorowane gdy ``max_original_features`` jest podane.
    random_state : int, default 42
        Seed RandomForest.

    Attributes set in ctx
    ---------------------
    ctx["leaf_encoder"] : RandomForestClassifier / RandomForestRegressor
        Fitowany RF używany do embeddingu.
    ctx["leaf_ohe"] : OneHotEncoder
        Fitowany enkoder one-hot liści (do transformacji nowych danych).
    ctx["leaf_feature_importances"] : ndarray
        Ważności cech z RF przed selekcją.
    ctx["leaf_selected_original"] : list of int or None
        Indeksy oryginalnych cech dołączonych do embeddingu.
    ctx["n_leaf_features"] : int
        Liczba cech z embeddingu liści.
    ctx["n_total_features"] : int
        Łączna liczba cech po połączeniu embeddingu z oryginalnymi.

    Notes
    -----
    Zalecana kolejność kroków:

    ``split → scale → leaf_encoding → fit → validate``

    Skalowanie przed embeddingiem nie jest konieczne (RF jest odporny na skalę),
    ale pomaga sieci neuronowej w dalszym kroku.

    Examples
    --------
    Domyślnie — 100 drzew, 32 liście, dołącz wszystkie oryginalne cechy:

    >>> step = LeafEncodingStep()

    Tylko embeddingi z liści, bez oryginalnych cech:

    >>> step = LeafEncodingStep(include_original=False)

    Top-10 oryginalnych cech + embeddingi:

    >>> step = LeafEncodingStep(max_original_features=10)

    Oryginalne cechy o ważności > 1% + embeddingi:

    >>> step = LeafEncodingStep(importance_threshold=0.01)

    Przez preset:

    >>> wf = SoftWorkflow.from_preset("leaf_embed")
    >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
    >>> print(result.ctx["n_leaf_features"])    # ile cech z embeddingu
    >>> print(result.ctx["n_total_features"])   # łącznie po połączeniu
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_leaf_nodes: int | None = 32,
        include_original: bool = True,
        max_original_features: int | None = None,
        importance_threshold: float | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.include_original = include_original
        self.max_original_features = max_original_features
        self.importance_threshold = importance_threshold
        self.random_state = random_state

    def _make_rf(self, y):
        from softnet.inference import TaskInferrer, TaskType
        task = TaskInferrer().infer(y)
        common = dict(
            n_estimators=self.n_estimators,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            n_jobs=-1,
        )
        if task.task == TaskType.REGRESSION:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**common)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**common)

    def _select_original_indices(self, importances) -> list[int] | None:
        import numpy as np
        if not self.include_original:
            return None
        if self.max_original_features is not None:
            idx = np.argsort(importances)[::-1][:self.max_original_features]
            return sorted(idx.tolist())
        if self.importance_threshold is not None:
            return [i for i, v in enumerate(importances) if v >= self.importance_threshold]
        # wszystkie oryginalne
        return list(range(len(importances)))

    def run(self, ctx: Context) -> Context:
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder

        if "X_train" not in ctx:
            raise ValueError(
                "LeafEncodingStep wymaga wcześniejszego SplitStep. "
                "Upewnij się, że 'split' jest przed 'leaf_encoding'."
            )

        X_train, X_test = ctx["X_train"], ctx["X_test"]
        y_train = ctx["y_train"]

        # 1. fit RandomForest
        rf = self._make_rf(y_train)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_

        # 2. leaf IDs: shape (n_samples, n_estimators)
        leaves_train = rf.apply(X_train)   # int32 leaf node IDs
        leaves_test  = rf.apply(X_test)

        # 3. one-hot encode liści — każde drzewo × każdy liść = 1 bit
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(leaves_train)
        embed_train = ohe.transform(leaves_train)
        embed_test  = ohe.transform(leaves_test)

        # 4. selekcja oryginalnych cech do dołączenia
        orig_idx = self._select_original_indices(importances)

        if orig_idx is not None:
            X_train_new = np.hstack([embed_train, X_train[:, orig_idx]])
            X_test_new  = np.hstack([embed_test,  X_test[:, orig_idx]])
        else:
            X_train_new = embed_train
            X_test_new  = embed_test

        ctx["X_train"] = X_train_new
        ctx["X_test"]  = X_test_new
        ctx["leaf_encoder"]              = rf
        ctx["leaf_ohe"]                  = ohe
        ctx["leaf_feature_importances"]  = importances
        ctx["leaf_selected_original"]    = orig_idx
        ctx["n_leaf_features"]           = embed_train.shape[1]
        ctx["n_total_features"]          = X_train_new.shape[1]
        ctx.setdefault("_transform_chain", []).append({
            "name": "leaf_encoding", "obj": rf, "ohe": ohe, "orig_idx": orig_idx,
        })

        orig_info = (
            f" + {len(orig_idx)} oryginalnych cech" if orig_idx is not None else ""
        )
        print(
            f"[soft-net] LeafEncoding: {X_train.shape[1]} cech  "
            f"→  {embed_train.shape[1]} leaf features{orig_info}  "
            f"=  {X_train_new.shape[1]} łącznie"
        )
        return ctx


# ── rejestr kroków ────────────────────────────────────────────────────────────

STEP_REGISTRY: dict[str, type] = {
    "split":             SplitStep,
    "scale":             ScaleStep,
    "pca":               PCAStep,
    "feature_selection": FeatureSelectionStep,
    "leaf_encoding":     LeafEncodingStep,
    "fit":               FitStep,
    "validate":          ValidateStep,
    "save":              SaveStep,
    "grid_search":       GridSearchStep,
    "randomized_search": RandomizedSearchStep,
    "voting":            VotingStep,
}
