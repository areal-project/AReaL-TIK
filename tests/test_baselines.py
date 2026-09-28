"""Numerical screening regressions; these tests require neither CUDA nor PyTorch."""

import unittest
from tools.baselines import numerical_error


class Output:
    shape = (1, 32, 1, 128)

    def __init__(self, error):
        self.error = error

    def float(self): return self
    def __sub__(self, other): return self
    def abs(self): return self
    def max(self): return self
    def item(self): return self.error


class BaselineTests(unittest.TestCase):
    def test_rejects_nonfinite_and_boundary_errors(self):
        for error in (float('nan'), float('inf'), float('-inf'), -0.1, 0.05, 0.1):
            with self.subTest(error=error), self.assertRaises(RuntimeError):
                numerical_error(Output(error), Output(0), 0.05)
        self.assertEqual(numerical_error(Output(0.004), Output(0), 0.005), 0.004)
        with self.assertRaises(RuntimeError):
            numerical_error(Output(0.004), Output(0), 0.003)

    def test_rejects_wrong_shape_and_invalid_threshold(self):
        wrong = Output(0)
        wrong.shape = (1,)
        with self.assertRaises(RuntimeError):
            numerical_error(wrong, Output(0), 0.05)
        for limit in (0, -1, float('nan'), float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                numerical_error(Output(0), Output(0), limit)
