"""
Task inference engine — detects problem type from data shape and labels.

Rules:
    y shape      | unique values   | dtype      → Task
    -------------|-----------------|------------|------------------
    (n,)         | 2               | int/bool   → binary
    (n,)         | >2              | int        → multiclass
    (n,)         | continuous      | float      → regression
    (n, k) k>1   | 0/1 only        | int        → multilabel
    (n, k) k>1   | continuous      | float      → multioutput_regression
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TaskType(Enum):
    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()
    REGRESSION = auto()
    MULTIOUTPUT_REGRESSION = auto()


@dataclass(frozen=True)
class TaskInfo:
    task: TaskType
    n_outputs: int          # output neurons
    n_classes: int | None   # None for regression

    def __str__(self) -> str:
        if self.n_classes:
            return f"{self.task.name}(classes={self.n_classes})"
        return f"{self.task.name}(outputs={self.n_outputs})"


class TaskInferrer:
    """Infer ML task type from target array y."""

    _FLOAT_DTYPES = (np.float16, np.float32, np.float64)

    def infer(self, y: np.ndarray) -> TaskInfo:
        y = np.asarray(y)

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            return self._infer_single_output(y.ravel())

        if y.ndim == 2:
            return self._infer_multi_output(y)

        raise ValueError(f"Unsupported y shape: {y.shape}. Expected 1-D or 2-D array.")

    def _infer_single_output(self, y: np.ndarray) -> TaskInfo:
        if y.dtype.type in self._FLOAT_DTYPES and not self._is_integer_valued(y):
            return TaskInfo(task=TaskType.REGRESSION, n_outputs=1, n_classes=None)

        unique = np.unique(y)
        n = len(unique)

        if n == 2:
            return TaskInfo(task=TaskType.BINARY, n_outputs=1, n_classes=2)

        return TaskInfo(task=TaskType.MULTICLASS, n_outputs=n, n_classes=n)

    def _infer_multi_output(self, y: np.ndarray) -> TaskInfo:
        k = y.shape[1]

        if y.dtype.type in self._FLOAT_DTYPES and not self._is_integer_valued(y):
            return TaskInfo(task=TaskType.MULTIOUTPUT_REGRESSION, n_outputs=k, n_classes=None)

        unique = np.unique(y)
        if set(unique).issubset({0, 1}):
            return TaskInfo(task=TaskType.MULTILABEL, n_outputs=k, n_classes=k)

        raise ValueError(
            f"Cannot infer task from 2-D y with dtype={y.dtype} and unique values={unique}. "
            "For multiclass provide 1-D y; for multilabel provide binary matrix."
        )

    @staticmethod
    def _is_integer_valued(y: np.ndarray) -> bool:
        return np.all(y == y.astype(int))
