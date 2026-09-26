from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from workloads.images import RetryLimitExceeded, present
from workloads.long_images import inference_parameters, training_parameters
from workloads.queue_images import run_plan
from zd3.constants import MODEL


def test_learning_rate_and_training_duration_survive_inference_metadata():
    args = SimpleNamespace(theta_plus_mv=.05, learning_rate_scale=.01, stimulus_ms=50.)
    training, variant = training_parameters(args, 3072)
    assert training.stimulus_ticks == 100
    assert training.rest_ticks == 300
    assert variant.potentiation_rate == pytest.approx(.0001)
    assert variant.depression_rate == pytest.approx(.000001)
    assert training.potentiation_rate == variant.potentiation_rate
    record = json.loads(json.dumps({'model': training.as_dict(), 'variant': variant.as_dict(),
        'inference_model': replace(training, stimulus_ms=350.).as_dict()}))
    inference, restored_variant = inference_parameters(record)
    assert inference.stimulus_ticks == 700
    assert inference.rest_ticks == 300
    assert inference.theta_plus_mv == .05
    assert restored_variant == variant
    # Existing theta10 manifests without the new inference field still work.
    record['model'] = inference.as_dict()
    del record['inference_model']
    assert inference_parameters(record) == (inference, variant)


def test_adaptation_does_not_accidentally_scale_stdp():
    args = SimpleNamespace(theta_plus_mv=2.5, learning_rate_scale=1., stimulus_ms=350.)
    training, variant = training_parameters(args, 784)
    assert training.theta_plus_mv == 50 * MODEL.theta_plus_mv
    assert variant.potentiation_rate == MODEL.potentiation_rate
    assert variant.depression_rate == MODEL.depression_rate
    assert training.stimulus_ms == MODEL.stimulus_ms


def test_retry_failure_keeps_all_attempts_and_blank_rests():
    class SilentNetwork:
        def __init__(self):
            self.operations = []

        def normalize(self):
            self.operations.append('normalize')

        def set_image(self, pixels, intensity):
            self.operations.append(('image', intensity))

        def run_stimulus(self):
            self.operations.append('stimulus')
            return np.array([1, 2])

        def run_rest(self):
            self.operations.append('blank_rest')

    network = SilentNetwork()
    with pytest.raises(RetryLimitExceeded, match='Retry intensity exceeded 3') as caught:
        present(network, np.zeros(784), MODEL, training=True, max_intensity=3)
    assert network.operations == ['normalize', ('image', 2.), 'stimulus', 'blank_rest',
                                  'normalize', ('image', 3.), 'stimulus', 'blank_rest']
    assert caught.value.details['attempts'] == 2
    assert caught.value.details['last_exc_spikes'] == 3
    assert caught.value.details['last_spike_counts'] == [1, 2]


def test_queue_orders_stages_and_continues_after_a_failed_case(tmp_path):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import json, pathlib, sys
output = pathlib.Path(sys.argv[1]); output.mkdir()
failed = sys.argv[2] == 'fail'
full = sys.argv[3] == 'full'
result = {'status': 'failed' if failed else 'complete', 'accepted_samples': 1}
(output / 'result.json').write_text(json.dumps(result))
if not failed:
    (output / 'evaluations.jsonl').write_text(json.dumps({'accepted_samples': 1, 'full_test': full, 'accuracy_percent': 50.0}) + '\\n')
sys.exit(1 if failed else 0)
''')
    root = tmp_path / 'queue'
    plan = {'artifact_root': str(root), 'source_root': str(tmp_path), 'source_sha256': {},
            'lock_path': str(tmp_path / 'queue.lock'), 'build_root': str(tmp_path / 'build'),
            'cwd': str(tmp_path), 'cases': []}
    for i, stage in enumerate(['cifar_pilot', 'cifar_pilot', 'fashion_long']):
        output = tmp_path / f'case_{i}'
        plan['cases'].append({'name': f'case_{i}', 'stage': stage, 'output': str(output),
            'log': str(root / 'logs' / f'case_{i}.log'), 'expected_accepted_samples': 1,
            'expected_full_test': stage == 'fashion_long',
            'command': [sys.executable, str(worker), str(output), 'fail' if i == 0 else 'pass',
                        'full' if stage == 'fashion_long' else 'probe']})
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    run_plan(plan_path)
    events = [json.loads(line) for line in (root / 'queue_events.jsonl').read_text().splitlines()]
    sequence = [(e['event'], e['case']) for e in events if 'case' in e]
    assert sequence == [(event, f'case_{i}') for i in range(3)
                        for event in ['case_started', 'case_finished']]
    pilot = json.loads((root / 'pilot_summary.json').read_text())
    assert [c['status'] for c in pilot['cases']] == ['failed', 'complete']
    result = json.loads((root / 'queue_result.json').read_text())
    assert result['completed_cases'] == 2 and result['failed_cases'] == 1
    with pytest.raises(FileExistsError):
        run_plan(plan_path)


def test_queue_refuses_overlapping_execution(tmp_path):
    import fcntl

    lock_path = tmp_path / 'queue.lock'
    plan = {'artifact_root': str(tmp_path / 'queue'), 'source_root': str(tmp_path),
            'source_sha256': {}, 'lock_path': str(lock_path), 'cases': []}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='holds the lock'):
            run_plan(plan_path)
