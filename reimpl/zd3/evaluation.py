from __future__ import annotations

import numpy as np


def assign_neurons(activity: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if activity.ndim != 2 or labels.shape != (activity.shape[0],):
        raise ValueError("activity and labels have incompatible shapes")
    assignments = np.full(activity.shape[1], -1, dtype=np.int16)
    maximum_rate = np.zeros(activity.shape[1], dtype=np.float64)
    for label in range(10):
        selected = labels == label
        if np.any(selected):
            rate = activity[selected].mean(axis=0)
            better = rate > maximum_rate
            assignments[better] = label
            maximum_rate[better] = rate[better]
    return assignments


def rank_classes(spike_counts: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    if spike_counts.ndim != 1 or assignments.shape != spike_counts.shape:
        raise ValueError("spike counts and assignments have incompatible shapes")
    scores = np.zeros(10, dtype=np.float64)
    for label in range(10):
        selected = assignments == label
        if np.any(selected):
            scores[label] = spike_counts[selected].mean()
    return np.argsort(scores, kind="stable")[::-1]


def simple_demo_accuracy(activity: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    assignments = assign_neurons(activity, labels)
    predictions = np.asarray(
        [rank_classes(row, assignments)[0] for row in activity], dtype=np.uint8
    )
    return {
        "accuracy_percent": 100.0 * float(np.mean(predictions == labels)),
        "assigned_neurons": int(np.count_nonzero(assignments >= 0)),
        "assignment_counts": [
            int(np.count_nonzero(assignments == label)) for label in range(10)
        ],
        "predictions": predictions,
        "assignments": assignments,
    }
