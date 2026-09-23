"""Exercise host normalization independently of CUDA or generated kernels."""
from types import SimpleNamespace
import unittest

import numpy as np

from backends.genn_backend import GeNNNetwork
from zd3.constants import MODEL


class NormalizationTest(unittest.TestCase):
    def network(self, weights):
        class Variable:
            def __init__(self, values):
                self.values = values.copy()
                self.pushes = 0

            def pull_from_device(self):
                pass

            def push_to_device(self):
                self.pushes += 1

        network = GeNNNetwork.__new__(GeNNNetwork)
        network.constants = MODEL
        network._feedforward_pre, network._feedforward_post = np.indices(weights.shape).reshape(2, -1)
        variable = Variable(weights.ravel())
        network.feedforward = SimpleNamespace(vars={"g": variable})
        network.variant = SimpleNamespace(normalization_weight_max_tolerance=0.02)
        network.weight_max_by_post = np.ones(MODEL.n_exc, dtype=np.float32)
        return network, variable

    def test_default_rejects_normalized_weight_above_cap(self):
        weights = np.ones((MODEL.n_input, MODEL.n_exc))
        weights[0, 0] = 100
        network, variable = self.network(weights)
        with self.assertRaisesRegex(RuntimeError, "postsynaptic wmax"):
            network.normalize()
        self.assertEqual(variable.pushes, 0)

    def test_opt_out_preserves_column_normalization_without_clipping(self):
        weights = np.ones((MODEL.n_input, MODEL.n_exc))
        weights[0, 0] = 100
        expected = weights * (MODEL.normalization_target / weights.sum(axis=0))[None, :]
        network, variable = self.network(weights)
        network.normalize(check_weight_bound=False)
        actual = variable.values.reshape(weights.shape)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=0), MODEL.normalization_target)
        self.assertGreater(actual[0, 0], network.weight_max_by_post[0])
        np.testing.assert_array_equal(network.weight_max_by_post, np.ones(MODEL.n_exc))
        self.assertEqual(variable.pushes, 1)

    def test_opt_out_still_rejects_zero_and_nonfinite_columns(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                weights = np.ones((MODEL.n_input, MODEL.n_exc))
                weights[:, 0] = value
                network, variable = self.network(weights)
                with self.assertRaisesRegex(ValueError, "non-finite or zero-sum"):
                    network.normalize(check_weight_bound=False)
                self.assertEqual(variable.pushes, 0)


if __name__ == "__main__":
    unittest.main()
