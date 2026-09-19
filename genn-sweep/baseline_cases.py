"""Frozen FP32 workload definitions for the cross-GPU baseline."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from zd3.constants import MODEL
from zd3.io import normalize_columns
from zd3.variants import NetworkVariant, get_variant


def mnist_cases() -> dict[str, NetworkVariant]:
    cases = {}
    for trace in (1, 2, 3):
        parent = get_variant("triplet-dense" if trace == 3 else "one-trace-dense")
        for topology, rate, tag in (
            ("dense", 1.0, "dense"),
            *((topology, rate, f"{topology}_{tag}")
              for topology in ("bernoulli", "fixed-fanout")
              for rate, tag in ((0.5, "0500"), (0.25, "0250"), (0.125, "0125"))),
        ):
            name = f"mnist_{trace}trace_{tag}"
            scale = (1.0 / rate) ** (1.0 if trace == 3 else 0.8)
            cases[name] = replace(
                parent, name=name,
                learning_rule="two-trace-power" if trace == 2 else parent.learning_rule,
                topology=topology, connection_rate=rate,
                weight_max=1.0 / rate,
                potentiation_rate=parent.potentiation_rate * scale,
                depression_rate=MODEL.depression_rate * scale,
                normalization_weight_max_tolerance=0.02 if rate < 1.0 else None,
            )
    return cases


def project_weights(weights: np.ndarray, mask: np.ndarray, weight_max: float) -> np.ndarray:
    """Preserve relative positive weights, subject to the column sum and cap."""
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("initial weights must be finite and nonnegative")
    projected = np.where(mask, weights, 0.0).astype(np.float64)
    normalize_columns(projected)
    for column in np.flatnonzero(projected.max(axis=0) > weight_max):
        values = projected[:, column].copy()
        capacity = np.count_nonzero(values) * weight_max
        if capacity < MODEL.normalization_target:
            raise ValueError("insufficient positive weight capacity for normalization")
        low, high = 0.0, 1.0
        while np.minimum(values * high, weight_max).sum() < MODEL.normalization_target:
            high *= 2.0
        for _ in range(64):
            middle = (low + high) / 2.0
            if np.minimum(values * middle, weight_max).sum() < MODEL.normalization_target:
                low = middle
            else:
                high = middle
        projected[:, column] = np.minimum(values * high, weight_max)
    return projected
