from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import MODEL, ModelConstants
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
    depression_rate: float = MODEL.depression_rate
    pre_weight_exponent: float = 0.2

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


def connectivity_mask(variant: NetworkVariant, *, model: ModelConstants = MODEL) -> np.ndarray:
    if variant.topology == "dense":
        return np.ones((model.n_input, model.n_exc), dtype=bool)
    if variant.topology in ("bernoulli", "fixed-fanout"):
        scores = np.random.RandomState(variant.connectivity_seed).uniform(
            size=(model.n_input, model.n_exc)
        )
        if variant.topology == "bernoulli":
            return scores < variant.connection_rate
        degree = round(model.n_exc * variant.connection_rate)
        if not 0 < degree <= model.n_exc:
            raise ValueError("fixed fan-out must select at least one target")
        # Ranking the same scores makes the masks nested across connection rates.
        targets = np.argsort(scores, axis=1, kind="stable")[:, :degree]
        mask = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(mask, targets, True, axis=1)
        return mask
    raise ValueError(f"unsupported topology: {variant.topology}")


def prepare_initial_weights(
    weights: np.ndarray, variant: NetworkVariant, *, weight_max_by_post: np.ndarray | None = None,
    model: ModelConstants = MODEL
) -> tuple[np.ndarray, np.ndarray]:
    if weights.shape != (model.n_input, model.n_exc):
        raise ValueError(f"unexpected feedforward shape {weights.shape}")
    mask = connectivity_mask(variant, model=model)
    prepared = np.where(mask, weights, 0.0).astype(np.float64, copy=False)
    normalize_columns(prepared, model.normalization_target)
    caps = variant.weight_max if weight_max_by_post is None else weight_max_by_post
    if np.any(prepared > caps * (1.0 + 1.0e-12)):
        raise ValueError(
            f"normalized initial weight exceeds wmax={variant.weight_max}"
        )
    return prepared, mask


def validate_checkpoint_topology(
    weights: np.ndarray, variant: NetworkVariant, *, model: ModelConstants = MODEL
) -> np.ndarray:
    mask = connectivity_mask(variant, model=model)
    if np.any(weights[~mask] != 0.0):
        raise ValueError(
            f"checkpoint contains weights outside {variant.name} structural mask"
        )
    return mask


def validate_normalized_weight_bound(
    weights: np.ndarray, variant: NetworkVariant, *, weight_max_by_post: np.ndarray | None = None
) -> None:
    tolerance = variant.normalization_weight_max_tolerance
    if tolerance is None:
        return
    if weight_max_by_post is not None:
        limits = np.asarray(weight_max_by_post, dtype=np.float64) * (1.0 + tolerance)
        violations = np.argwhere(weights > limits[None, :])
        if len(violations):
            pre, post = violations[0]
            raise RuntimeError(
                f"normalization exceeded postsynaptic wmax tolerance: pre={pre} post={post} "
                f"weight={weights[pre, post]:.9f} wmax={weight_max_by_post[post]:.9f} "
                f"limit={limits[post]:.9f}"
            )
        return
    maximum = float(np.max(weights))
    limit = variant.weight_max * (1.0 + tolerance)
    if maximum > limit:
        raise RuntimeError(
            f"normalization exceeded wmax tolerance: max={maximum:.9f} "
            f"limit={limit:.9f}"
        )
