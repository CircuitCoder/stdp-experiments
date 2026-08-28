from __future__ import annotations

import numpy as np

from locality.analysis import (
    event_series,
    histogram_concentration,
    morans_i_4_neighbor,
    scan_brunel_graph,
)


def test_event_series_applies_delay_and_window_boundaries() -> None:
    series = event_series(
        np.asarray([0.0, 0.1, 0.2]),
        np.asarray([0, 1, 0]),
        np.asarray([4, 7]),
        dt_ms=0.1,
        start_ms=0.1,
        duration_ticks=2,
        delay_ms=0.1,
    )
    np.testing.assert_array_equal(series, [4, 7])


def test_histogram_concentration_handles_uniform_and_concentrated_values() -> None:
    uniform = histogram_concentration(np.asarray([0, 4], dtype=np.uint64))
    assert uniform["gini"] == 0.0
    concentrated = histogram_concentration(np.asarray([3, 0, 1], dtype=np.uint64))
    assert concentrated["gini"] == 0.75


def test_scan_brunel_graph_counts_multapses_and_community_mass() -> None:
    row_lengths = np.asarray([2, 1, 2], dtype=np.uint32)
    raw_posts = np.asarray([1, 2, 99, 0, 99, 99, 0, 1, 99], dtype=np.uint32)
    summary, arrays = scan_brunel_graph(
        row_lengths,
        raw_posts,
        3,
        np.asarray([2, 0, 1], dtype=np.uint32),
        np.asarray([1, 3, 0], dtype=np.uint32),
        np.asarray([0, 0, 1], dtype=np.int16),
        rank_blocks=3,
    )
    assert summary["structural_edges_including_multapses"] == 5
    assert summary["total_update_attempts"] == 14
    assert summary["within_community_mass_fraction"] == 6 / 14
    np.testing.assert_array_equal(summary["edge_frequency_histogram"], [0, 1, 2, 0, 1, 1])
    np.testing.assert_array_equal(arrays["out_strength"], [7, 1, 6])
    np.testing.assert_array_equal(arrays["in_strength"], [3, 9, 2])


def test_morans_i_rejects_constant_image_and_detects_spatial_clustering() -> None:
    assert morans_i_4_neighbor(np.ones((4, 4))) is None
    clustered = np.zeros((4, 4))
    clustered[:2, :2] = 1.0
    assert morans_i_4_neighbor(clustered) > 0.0
