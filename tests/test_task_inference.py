"""Tests for TaskInferrer — core rules engine."""

import numpy as np
import pytest

from softnet.inference import TaskInferrer, TaskType


inferrer = TaskInferrer()


class TestBinary:
    def test_detects_binary_from_0_1(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        info = inferrer.infer(y)
        assert info.task == TaskType.BINARY
        assert info.n_outputs == 1
        assert info.n_classes == 2

    def test_detects_binary_from_string_like_int(self):
        y = np.array([0, 1, 1, 0])
        info = inferrer.infer(y)
        assert info.task == TaskType.BINARY


class TestMulticlass:
    def test_detects_multiclass_3_classes(self):
        y = np.array([0, 1, 2, 1, 0, 2])
        info = inferrer.infer(y)
        assert info.task == TaskType.MULTICLASS
        assert info.n_classes == 3
        assert info.n_outputs == 3

    def test_detects_multiclass_5_classes(self):
        y = np.arange(100) % 5
        info = inferrer.infer(y)
        assert info.task == TaskType.MULTICLASS
        assert info.n_classes == 5


class TestRegression:
    def test_detects_regression_from_float(self):
        y = np.array([1.2, 3.4, 5.6, 7.8])
        info = inferrer.infer(y)
        assert info.task == TaskType.REGRESSION
        assert info.n_outputs == 1
        assert info.n_classes is None

    def test_integer_valued_float_is_classification(self):
        y = np.array([0.0, 1.0, 2.0])
        info = inferrer.infer(y)
        assert info.task == TaskType.MULTICLASS


class TestMultilabel:
    def test_detects_multilabel_from_binary_matrix(self):
        y = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
        info = inferrer.infer(y)
        assert info.task == TaskType.MULTILABEL
        assert info.n_outputs == 3


class TestMultioutputRegression:
    def test_detects_multioutput_regression(self):
        y = np.random.rand(100, 3)
        info = inferrer.infer(y)
        assert info.task == TaskType.MULTIOUTPUT_REGRESSION
        assert info.n_outputs == 3
