"""
soft-net workflows — presety cykli ML.

Dostępne presety: basic, scaled, scaled_save, grid_search,
                  randomized_search, voting, full, full_search

Quick start
-----------
>>> from softnet.workflows import SoftWorkflow, list_workflows
>>> from softnet import SoftClassifier

>>> list_workflows()                          # podejrzyj dostępne presety

>>> wf = SoftWorkflow.from_preset("scaled")
>>> result = wf.run(X, y, estimator=SoftClassifier.from_preset("medium"))
>>> print(result.score)
>>> print(result.report)
"""

from softnet.workflows.workflow import SoftWorkflow, WorkflowResult
from softnet.workflows.registry import (
    WorkflowPreset,
    list_workflows,
    get_workflow,
    register_workflow,
    load_workflows_from_toml,
)
from softnet.workflows.steps import (
    SplitStep,
    ScaleStep,
    PCAStep,
    FeatureSelectionStep,
    FitStep,
    ValidateStep,
    SaveStep,
    GridSearchStep,
    RandomizedSearchStep,
    VotingStep,
)

__all__ = [
    # główne klasy
    "SoftWorkflow",
    "WorkflowResult",
    # preset API
    "WorkflowPreset",
    "list_workflows",
    "get_workflow",
    "register_workflow",
    "load_workflows_from_toml",
    # kroki
    "SplitStep",
    "ScaleStep",
    "PCAStep",
    "FeatureSelectionStep",
    "FitStep",
    "ValidateStep",
    "SaveStep",
    "GridSearchStep",
    "RandomizedSearchStep",
    "VotingStep",
]
