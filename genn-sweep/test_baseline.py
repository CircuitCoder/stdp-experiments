import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reimpl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_cases import mnist_cases, project_weights
from zd3.variants import connectivity_mask


def test_matrix_degrees_and_nested_masks():
    cases = mnist_cases()
    assert len(cases) == 21
    for trace in (1, 2, 3):
        previous = np.ones((784, 400), dtype=bool)
        for tag, degree in (("0500", 200), ("0250", 100), ("0125", 50)):
            mask = connectivity_mask(cases[f"mnist_{trace}trace_fixed-fanout_{tag}"])
            np.testing.assert_array_equal(mask.sum(axis=1), degree)
            assert np.all(~mask | previous)
            assert not np.all(mask.sum(axis=0) == round(784 * degree / 400))
            previous = mask
    for name, variant in cases.items():
        if "1trace" in name:
            for trace in (2, 3):
                np.testing.assert_array_equal(connectivity_mask(variant),
                    connectivity_mask(cases[name.replace("1trace", f"{trace}trace")]))


def test_projection_preserves_structure_sum_and_bound():
    variant = mnist_cases()["mnist_2trace_fixed-fanout_0125"]
    mask = connectivity_mask(variant)
    weights = np.random.RandomState(1).lognormal(size=mask.shape)
    projected = project_weights(weights, mask, 8.0)
    np.testing.assert_allclose(projected.sum(axis=0), 78.0, atol=1e-10)
    assert np.all(projected[~mask] == 0.0)
    assert projected.max() <= 8.0
    with pytest.raises(ValueError):
        project_weights(np.zeros_like(weights), mask, 8.0)
