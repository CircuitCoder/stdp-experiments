"""Prevent the throughput/latency reporting error from recurring."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cpu_latency_summary', Path(__file__).with_name('summarize.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def records():
    worker = {'wall_seconds': 2.5, 'simulation_steps': 100000,
              'us_per_step': 25.0, 'cpu_affinity': [9]}
    batch = {'metric': 'single_network_latency', 'replicas': 1,
             'simulation_threads_per_replica': 1, 'us_per_step': 25.0,
             'batch_wall_seconds': 2.6, 'workers': [worker]}
    measurement = {'metric': 'single_network_latency', 'replicas': 1,
                   'repetitions': 1, 'median_us_per_step': 25.0}
    return measurement, [batch]


def test_uses_network_timer_excluding_parent_rendezvous():
    measurement, batches = records()
    assert report.latency_samples(measurement, batches) == [25.0]


def test_rejects_historical_throughput_summary():
    with pytest.raises(ValueError, match='throughput'):
        report.latency_samples({'replicas': 32, 'aggregate_us_per_step_median': 1.96}, [])


@pytest.mark.parametrize('failure', ['multiple_networks', 'divided_by_replicas', 'incomplete', 'unpinned', 'summary_mismatch'])
def test_rejects_incompatible_or_miscomputed_results(failure):
    measurement, batches = records()
    if failure == 'multiple_networks':
        batches[0]['workers'].append(deepcopy(batches[0]['workers'][0]))
    elif failure == 'divided_by_replicas':
        batches[0]['us_per_step'] /= 32
    elif failure == 'incomplete':
        measurement['repetitions'] = 5
    elif failure == 'unpinned':
        batches[0]['workers'][0]['cpu_affinity'] = [8, 9]
    else:
        measurement['median_us_per_step'] = 1.96
    with pytest.raises(ValueError):
        report.latency_samples(measurement, batches)
