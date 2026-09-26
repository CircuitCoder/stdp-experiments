"""Deterministic generated-kernel check for long image-training timestamps."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'reimpl'), str(ROOT / 'genn-sweep')]
from backends.genn_backend import GeNNNetwork
from baseline_cases import mnist_cases
from zd3.constants import MODEL


def run(args):
    psutil.cpu_count = lambda logical=True: 4
    variant = mnist_cases()['mnist_3trace_dense']
    constants = replace(MODEL, n_input=2, n_exc=2, n_inh=2)
    pre = {5, 31, 35, 80}
    post = {10, 35, 55, 95}
    offsets = [0] if args.mode == 'float' else [0, 2**24 + 11, 2**28 + 3, 2**31 + 1, 2**32 - 200]
    rows = []
    for offset in offsets:
        net = GeNNNetwork(weights=np.full((2, 2), .4), theta_mv=np.full(2, 20.),
            plasticity=True, inhibition=17., seed=17, backend=args.backend,
            build_path=args.output, variant=variant, structural_mask=np.ones((2, 2), bool),
            precision='float', parallelism='postsynaptic', num_threads_per_spike=1,
            timing_enabled=False, reuse_build=args.output if args.output.exists() else None,
            constants=constants, integer_timestamps=args.mode == 'integer')
        try:
            net.model.timestep = offset
            if args.mode == 'integer':
                for pop in (net.inputs, net.exc):
                    pop.vars['tick'].view[:] = offset
                    pop.vars['tick'].push_to_device()
            for tick in range(120):
                rates = net.inputs.vars['rateHz']
                rates.view[:] = [2000. if tick in pre else 0., 0.]
                rates.push_to_device()
                # Suppress spontaneous spikes and schedule E0 exactly.
                for name, value in [('V', [21. if tick in post else -65., -65.]), ('refrac', 0)]:
                    var = net.exc.vars[name]
                    var.view[:] = value
                    var.push_to_device()
                net.model.step_time()
            counters = net.event_counters()
            assert counters['input_spikes'] == len(pre)
            assert counters['excitatory_spikes'] == len(post)
            weight = float(net.weights()[0, 0])
            expected = float(np.float32(.4))
            last_pre = last_post = None
            for tick in range(120):
                if tick in pre:
                    if last_post is not None:
                        expected -= variant.depression_rate * np.exp(-(tick - last_post) * .5 / 20)
                    last_pre = tick
                if tick in post:
                    if last_pre is not None and last_post is not None:
                        expected += variant.potentiation_rate * np.exp(-(tick - last_pre) * .5 / 20) * np.exp(-(tick - last_post) * .5 / 40)
                    last_post = tick
            np.testing.assert_allclose(weight, expected, rtol=0, atol=1e-7)
            if rows:
                assert weight == rows[0]['weight']
            rows.append({'offset': offset, 'weight': weight, 'expected': expected})
        finally:
            net.close()
    print(json.dumps(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True)
    parser.add_argument('--mode', choices=['float', 'integer'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
