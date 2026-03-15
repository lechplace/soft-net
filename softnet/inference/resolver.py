"""
Config resolver — maps TaskInfo → ModelConfig (loss, activation, metrics, optimizer).

This is the "rules engine" that encodes domain knowledge so users don't have to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .task import TaskInfo, TaskType


@dataclass
class ModelConfig:
    """Fully resolved Keras model configuration."""

    loss: str
    output_activation: str
    output_units: int
    metrics: list[str]
    optimizer: str = "adam"
    learning_rate: float = 1e-3

    # human-readable explanation — useful for debugging / transparency
    rationale: str = ""


_RULES: dict[TaskType, dict] = {
    TaskType.BINARY: {
        "loss": "binary_crossentropy",
        "output_activation": "sigmoid",
        "metrics": ["accuracy", "AUC"],
        "rationale": (
            "Binary classification: sigmoid squashes output to [0,1]; "
            "binary_crossentropy is the canonical loss for two-class problems."
        ),
    },
    TaskType.MULTICLASS: {
        "loss": "sparse_categorical_crossentropy",
        "output_activation": "softmax",
        "metrics": ["accuracy"],
        "rationale": (
            "Multiclass classification: softmax ensures outputs sum to 1 (mutually exclusive); "
            "sparse_categorical_crossentropy works with integer labels (no one-hot needed)."
        ),
    },
    TaskType.MULTILABEL: {
        "loss": "binary_crossentropy",
        "output_activation": "sigmoid",
        "metrics": ["accuracy"],
        "rationale": (
            "Multilabel classification: each label is independent, so sigmoid per neuron; "
            "binary_crossentropy treats each label as a separate binary problem."
        ),
    },
    TaskType.REGRESSION: {
        "loss": "mean_squared_error",
        "output_activation": "linear",
        "metrics": ["mean_absolute_error"],
        "rationale": (
            "Regression: linear activation to allow any real-valued output; "
            "MSE penalises large errors strongly."
        ),
    },
    TaskType.MULTIOUTPUT_REGRESSION: {
        "loss": "mean_squared_error",
        "output_activation": "linear",
        "metrics": ["mean_absolute_error"],
        "rationale": (
            "Multi-output regression: same as regression but with k output neurons."
        ),
    },
}


class ConfigResolver:
    """Resolve a complete ModelConfig from a TaskInfo."""

    def resolve(self, task_info: TaskInfo) -> ModelConfig:
        rule = _RULES[task_info.task]

        return ModelConfig(
            loss=rule["loss"],
            output_activation=rule["output_activation"],
            output_units=task_info.n_outputs,
            metrics=list(rule["metrics"]),
            rationale=rule["rationale"],
        )
