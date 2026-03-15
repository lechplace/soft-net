"""Tests for ConfigResolver — smart defaults mapping."""

import pytest

from softnet.inference import TaskInferrer, ConfigResolver, TaskType


inferrer = TaskInferrer()
resolver = ConfigResolver()


def resolve(y_factory):
    import numpy as np
    y = y_factory()
    task_info = inferrer.infer(y)
    return resolver.resolve(task_info)


class TestBinaryDefaults:
    def test_loss(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 0, 1]))
        assert cfg.loss == "binary_crossentropy"

    def test_activation(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 0, 1]))
        assert cfg.output_activation == "sigmoid"

    def test_output_units(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 0, 1]))
        assert cfg.output_units == 1


class TestMulticlassDefaults:
    def test_loss(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 2, 1, 0]))
        assert cfg.loss == "sparse_categorical_crossentropy"

    def test_activation(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 2, 1, 0]))
        assert cfg.output_activation == "softmax"

    def test_output_units_equals_n_classes(self):
        import numpy as np
        cfg = resolve(lambda: np.array([0, 1, 2, 3, 4]))
        assert cfg.output_units == 5


class TestRegressionDefaults:
    def test_loss(self):
        import numpy as np
        cfg = resolve(lambda: np.random.rand(50))
        assert cfg.loss == "mean_squared_error"

    def test_linear_activation(self):
        import numpy as np
        cfg = resolve(lambda: np.random.rand(50))
        assert cfg.output_activation == "linear"


class TestRationaleIsNotEmpty:
    def test_all_tasks_have_rationale(self):
        import numpy as np
        datasets = [
            np.array([0, 1, 0, 1]),           # binary
            np.array([0, 1, 2]),               # multiclass
            np.random.rand(50),                # regression
            np.array([[0, 1], [1, 0]]),        # multilabel
            np.random.rand(50, 2),             # multioutput reg
        ]
        for y in datasets:
            info = inferrer.infer(y)
            cfg = resolver.resolve(info)
            assert len(cfg.rationale) > 0, f"Missing rationale for {info.task}"
