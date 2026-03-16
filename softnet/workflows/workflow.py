"""
SoftWorkflow — orkiestrator kroków ML.

Łączy kroki (split, scale, fit, validate, save, …) w spójny pipeline
z automatycznym zarządzaniem kontekstem danych.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── WorkflowResult ────────────────────────────────────────────────────────────

@dataclass
class WorkflowResult:
    """
    Wynik wykonania workflow.

    Attributes
    ----------
    score : float
        Główna metryka: accuracy (klasyfikacja) lub R² (regresja).
    report : str
        Pełny raport tekstowy z metrykami.
    model : Any
        Wytrenowany estimator.
    scaler : Any or None
        Wytrenowany scaler (jeśli użyto ``ScaleStep``), inaczej ``None``.
    best_params : dict or None
        Najlepsze hiperparametry (jeśli użyto ``GridSearchStep``
        lub ``RandomizedSearchStep``), inaczej ``None``.
    saved_path : str or None
        Ścieżka zapisanego modelu (jeśli użyto ``SaveStep``), inaczej ``None``.
    ctx : dict
        Pełny kontekst workflow — surowe dane, pośrednie wyniki, historia treningu.

    Examples
    --------
    >>> result = workflow.run(X, y, estimator=clf)
    >>> print(result.score)
    0.9733
    >>> print(result.report)
    >>> result.model.summary()
    """

    score: float = 0.0
    report: str = ""
    model: Any = None
    scaler: Any = None
    best_params: dict | None = None
    saved_path: str | None = None
    ctx: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"score={self.score:.4f}"]
        if self.best_params:
            parts.append(f"best_params={self.best_params}")
        if self.saved_path:
            parts.append(f"saved='{self.saved_path}'")
        return f"WorkflowResult({', '.join(parts)})"


# ── SoftWorkflow ──────────────────────────────────────────────────────────────

class SoftWorkflow:
    """
    Orkiestrator kroków ML — wykonuje sekwencję kroków na danych.

    Każdy krok operuje na wspólnym słowniku kontekstu (X_train, y_train, …)
    i przekazuje go do następnego kroku.

    Parameters
    ----------
    steps : list
        Lista instancji kroków (np. ``SplitStep()``, ``ScaleStep()``, …).
    estimator : sklearn-compatible estimator, optional
        Domyślny estimator. Może być nadpisany w ``run()``.

    Examples
    --------
    Ręczna budowa workflow:

    >>> from softnet.workflows.steps import SplitStep, ScaleStep, FitStep, ValidateStep
    >>> from softnet import SoftClassifier
    >>>
    >>> wf = SoftWorkflow(
    ...     steps=[SplitStep(), ScaleStep(), FitStep(), ValidateStep()],
    ...     estimator=SoftClassifier.from_preset("medium"),
    ... )
    >>> result = wf.run(X, y)
    >>> print(result.score)

    Użycie presetu:

    >>> wf = SoftWorkflow.from_preset("scaled")
    >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium", epochs=100))
    >>> print(result.score)

    Voting ensemble:

    >>> wf = SoftWorkflow.from_preset("voting")
    >>> result = wf.run(X, y)

    Grid search:

    >>> from softnet.workflows.steps import GridSearchStep
    >>> wf = SoftWorkflow.from_preset(
    ...     "grid_search",
    ...     step_overrides={"grid_search": GridSearchStep(
    ...         param_grid={"epochs": [50, 100, 200], "dropout": [0.1, 0.3]},
    ...         cv=3,
    ...     )},
    ... )
    >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
    >>> print(result.best_params)

    See Also
    --------
    list_workflows : Wypisz dostępne presety workflow.
    WorkflowResult : Klasa wynikowa z metrykami i modelem.
    """

    def __init__(self, steps: list, estimator=None):
        self.steps = steps
        self.estimator = estimator

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        name: str,
        estimator=None,
        step_overrides: dict | None = None,
        **step_params,
    ) -> "SoftWorkflow":
        """
        Utwórz workflow z presetu.

        Parameters
        ----------
        name : str
            Nazwa presetu. Dostępne → ``list_workflows()``.
        estimator : sklearn-compatible estimator, optional
            Estimator do użycia w ``FitStep``. Może być podany później w ``run()``.
        step_overrides : dict of {step_name: step_instance}, optional
            Podmień konkretny krok własną instancją.
            Np. ``{"scale": ScaleStep(method="robust")}``.
        **step_params
            Dodatkowe parametry przekazywane do kroków (rozszerzenie domyślnych).

        Returns
        -------
        SoftWorkflow

        Examples
        --------
        >>> wf = SoftWorkflow.from_preset("scaled")
        >>> wf = SoftWorkflow.from_preset("grid_search",
        ...     step_overrides={"split": SplitStep(test_size=0.15)})
        """
        from softnet.workflows.registry import get_workflow
        from softnet.workflows.steps import STEP_REGISTRY

        preset = get_workflow(name)
        step_overrides = step_overrides or {}
        steps = []

        for step_name in preset.steps:
            if step_name in step_overrides:
                steps.append(step_overrides[step_name])
            else:
                StepCls = STEP_REGISTRY[step_name]
                params = dict(preset.step_params.get(step_name, {}))
                params.update(step_params.get(step_name, {}))
                # filtruj tylko parametry akceptowane przez __init__
                import inspect
                valid = inspect.signature(StepCls.__init__).parameters
                filtered = {k: v for k, v in params.items() if k in valid}
                steps.append(StepCls(**filtered))

        return cls(steps=steps, estimator=estimator)

    # ------------------------------------------------------------------
    # główna metoda
    # ------------------------------------------------------------------

    def run(self, X, y, estimator=None) -> WorkflowResult:
        """
        Wykonaj workflow na danych.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Macierz cech.
        y : array-like of shape (n_samples,)
            Wektor targetów.
        estimator : sklearn-compatible estimator, optional
            Nadpisuje estimator zdefiniowany w konstruktorze.

        Returns
        -------
        WorkflowResult
            Obiekt z wynikami: ``score``, ``report``, ``model``, ``scaler``,
            ``best_params``, ``saved_path``, ``ctx``.

        Examples
        --------
        >>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
        >>> print(result.score)
        >>> print(result.report)
        >>> result.model.summary()
        """
        import numpy as np
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y)

        ctx: dict[str, Any] = {
            "X": X,
            "y": y,
            "estimator": estimator or self.estimator,
        }

        step_names = [type(s).__name__ for s in self.steps]
        print(f"[soft-net] Workflow: {' → '.join(step_names)}")

        for step in self.steps:
            ctx = step.run(ctx)

        validation = ctx.get("validation", {})
        return WorkflowResult(
            score=validation.get("score", 0.0),
            report=validation.get("report", ""),
            model=ctx.get("fitted_estimator"),
            scaler=ctx.get("scaler"),
            best_params=ctx.get("best_params"),
            saved_path=ctx.get("saved_path"),
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self.steps]
        return f"SoftWorkflow([{', '.join(step_names)}])"
