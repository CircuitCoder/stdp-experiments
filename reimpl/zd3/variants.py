from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import MODEL
from .io import normalize_columns


@dataclass(frozen=True)
class NetworkVariant:
    name: str
    learning_rule: str
    topology: str
    connection_rate: float
    connectivity_seed: int
    weight_max: float
    potentiation_rate: float
    pre_trace_target: float = 0.4
    post_weight_exponent: float = 0.2
    normalization_weight_max_tolerance: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


VARIANTS = {
    "triplet-dense": NetworkVariant(
        name="triplet-dense",
        learning_rule="three-trace",
        topology="dense",
        connection_rate=1.0,
        connectivity_seed=20260723,
        weight_max=1.0,
        potentiation_rate=MODEL.potentiation_rate,
    ),
    "one-trace-dense": NetworkVariant(
        name="one-trace-dense",
        learning_rule="one-trace-power",
        topology="dense",
        connection_rate=1.0,
        connectivity_seed=20260723,
        weight_max=1.0,
        potentiation_rate=0.0005,
    ),
    "one-trace-bernoulli-0125": NetworkVariant(
        name="one-trace-bernoulli-0125",
        learning_rule="one-trace-power",
        topology="bernoulli",
        connection_rate=0.125,
        connectivity_seed=20260723,
        weight_max=8.0,
        potentiation_rate=0.002639015821545789,
        normalization_weight_max_tolerance=0.02,
    ),
}


def get_variant(name: str) -> NetworkVariant:
    try:
        return VARIANTS[name]
    except KeyError as error:
        raise ValueError(f"unknown network variant: {name}") from error


def connectivity_mask(variant: NetworkVariant) -> np.ndarray:
    if variant.topology == "dense":
        return np.ones((MODEL.n_input, MODEL.n_exc), dtype=bool)
    if variant.topology == "bernoulli":
        scores = np.random.RandomState(variant.connectivity_seed).uniform(
            size=(MODEL.n_input, MODEL.n_exc)
        )
        return scores < variant.connection_rate
    raise ValueError(f"unsupported topology: {variant.topology}")


def prepare_initial_weights(
    weights: np.ndarray, variant: NetworkVariant
) -> tuple[np.ndarray, np.ndarray]:
    if weights.shape != (MODEL.n_input, MODEL.n_exc):
        raise ValueError(f"unexpected feedforward shape {weights.shape}")
    mask = connectivity_mask(variant)
    prepared = np.where(mask, weights, 0.0).astype(np.float64, copy=False)
    normalize_columns(prepared)
    if np.max(prepared[mask]) > variant.weight_max * (1.0 + 1.0e-12):
        raise ValueError(
            f"normalized initial weight exceeds wmax={variant.weight_max}"
        )
    return prepared, mask


def validate_checkpoint_topology(
    weights: np.ndarray, variant: NetworkVariant
) -> np.ndarray:
    mask = connectivity_mask(variant)
    if np.any(weights[~mask] != 0.0):
        raise ValueError(
            f"checkpoint contains weights outside {variant.name} structural mask"
        )
    return mask


def validate_normalized_weight_bound(
    weights: np.ndarray, variant: NetworkVariant
) -> None:
    tolerance = variant.normalization_weight_max_tolerance
    if tolerance is None:
        return
    maximum = float(np.max(weights))
    limit = variant.weight_max * (1.0 + tolerance)
    if maximum > limit:
        raise RuntimeError(
            f"normalization exceeded wmax tolerance: max={maximum:.9f} "
            f"limit={limit:.9f}"
        )
