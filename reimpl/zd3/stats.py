from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntervalStats:
    accepted: int = 0
    attempts: int = 0
    retries: int = 0
    spikes: int = 0
    active_neurons: int = 0
    intensity_sum: float = 0.0
    intensity_max: float = 2.0

    def record_attempt(self, retry: bool) -> None:
        self.attempts += 1
        self.retries += int(retry)

    def record_accepted(self, counts: np.ndarray, intensity: float) -> None:
        self.accepted += 1
        self.spikes += int(counts.sum())
        self.active_neurons += int(np.count_nonzero(counts))
        self.intensity_sum += intensity
        self.intensity_max = max(self.intensity_max, intensity)

    def format(
        self,
        *,
        accepted: int,
        weights: np.ndarray,
        theta_mv: np.ndarray,
        interval: int,
        backend: str,
        weight_max: float = 1.0,
        runtime: dict[str, float] | None = None,
    ) -> str:
        del interval
        column_sums = weights.sum(axis=0, dtype=np.float64)
        retry_fraction = self.retries / self.attempts if self.attempts else 0.0
        denominator = self.accepted if self.accepted else 1
        text = (
            f"DIAGNOSTICS backend={backend} accepted={accepted} "
            f"attempts={self.attempts} retries={self.retries} "
            f"retry_fraction={retry_fraction:.6f} "
            f"spikes_mean={self.spikes / denominator:.6f} "
            f"active_mean={self.active_neurons / denominator:.6f} "
            f"intensity_mean={self.intensity_sum / denominator:.6f} "
            f"intensity_max={self.intensity_max:.6f} "
            f"theta_mean_mv={theta_mv.mean():.6f} "
            f"theta_min_mv={theta_mv.min():.6f} "
            f"theta_max_mv={theta_mv.max():.6f} "
            f"weight_mean={weights.mean(dtype=np.float64):.9f} "
            f"weight_min={weights.min():.9f} weight_max={weights.max():.9f} "
            f"weight_zero_fraction={np.mean(weights <= 0.0):.9f} "
            f"weight_saturated_fraction={np.mean(weights >= weight_max):.9f} "
            f"colsum_mean={column_sums.mean():.9f} "
            f"colsum_min={column_sums.min():.9f} "
            f"colsum_max={column_sums.max():.9f}"
        )
        if runtime:
            text += " " + " ".join(
                f"{name}={value:.6f}" for name, value in runtime.items()
            )
        return text
