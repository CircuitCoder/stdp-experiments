#!/usr/bin/env python3
"""Full-size Morrison 2007 network with the requested zero transport delay."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'brunel')]
from ports.common import Model, Rule, DT_MS
from ports.genn_port import GeNNBrunel, stdp_timestamp_metadata
from workloads.provenance import manifest, write_json, command


@dataclass(frozen=True)
class MorrisonModel(Model):
    tau_syn_ms: float = 0.33
    je_pa: float = 45.61
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0

    @property
    def external_rate_hz(self):
        return 9000 * 2.32 * self.external_rate_scale

    def as_dict(self):
        result = super().as_dict()
        result.update({key: getattr(self, key) for key in
                       ('tau_syn_ms', 'je_pa', 'tau_plus_ms', 'tau_minus_ms')})
        result['family'] = 'morrison-2007-zero-delay-v1'
        result['effective_delivery_ms'] = DT_MS
        result['stdp_tie_order'] = 'nest_causal_boundary'
        result.update(stdp_timestamp_metadata())
        result['parent_delay_ms'] = 1.5
        return result


def make_morrison():
    return MorrisonModel(
        rule=Rule('morrison', 5.0, 1.0, 0.1, 0.1057, 0.4, 1.0, None),
        network_scale=10.0, indegree_scale=1.0,
        ne=90000, ni=22500, ce=9000, ci=2250,
        delay_ms=0.0, recurrent_delivery_scale=1.0, external_rate_scale=1.0)


def inspect_state(network, spec):
    result = {}
    for name, population in [('e', network.exc), ('i', network.inh)]:
        for field in ('V', 'Iex', 'Iin', 'dIex', 'dIin'):
            variable = population.vars[field]
            variable.pull_from_device()
            values = np.asarray(variable.view)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f'Non-finite {name}.{field}')
            result[f'{name}_{field}_min'] = float(values.min())
            result[f'{name}_{field}_max'] = float(values.max())
    return result


def sample_weights(network, spec, count=100000):
    # Sample real edges from padded GeNN storage without materializing a sorted
    # float64 copy of all 810M weights and the entire connectivity matrix.
    network.ee._row_lengths.pull_from_device()
    lengths = np.asarray(network.ee._row_lengths.view, dtype=np.int64)
    cumulative = np.cumsum(lengths)
    if int(cumulative[-1]) != spec.plastic_synapses:
        raise RuntimeError('Unexpected E-E connectivity count')
    logical = np.linspace(0, cumulative[-1] - 1, count, dtype=np.int64)
    rows = np.searchsorted(cumulative, logical, side='right')
    offsets = logical - np.r_[0, cumulative][rows]
    variable = network.ee.vars['g']
    variable.pull_from_device()
    # PyGeNN 5.4 deliberately disallows public sparse views. Its backing array
    # is padded row-major; use it here to avoid an additional 3.24 GB copy.
    values = np.asarray(variable._view).reshape(spec.ne, -1)[rows, offsets].astype(np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.max() > 1000:
        raise RuntimeError('Invalid/exploding sampled weights')
    return {'sample_count': int(values.size), 'min': float(values.min()),
            'max': float(values.max()), 'mean': float(values.mean()),
            'std': float(values.std()), 'zero_fraction': float(np.mean(values == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sim-ms', type=float, default=1000)
    parser.add_argument('--chunk-ms', type=float, default=100)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--abort-rate-hz', type=float, default=100)
    args = parser.parse_args()
    if args.seed <= 0 or args.sim_ms <= 0 or args.chunk_ms <= 0:
        parser.error('seed and durations must be positive')
    for value in (args.sim_ms, args.chunk_ms):
        if not np.isclose(value / DT_MS, round(value / DT_MS)):
            parser.error('durations must be integral timesteps')
    gpu = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'])
    if gpu['returncode'] or not any(
            ('A800' in line or 'H800' in line) and int(line.split(',')[1]) >= 75000
            for line in gpu['stdout'].splitlines()):
        raise RuntimeError('This full-size pilot requires an allocated 80 GB A800/H800')
    args.output.mkdir(parents=True, exist_ok=False)
    spec = make_morrison()
    record = manifest(args) | {'model': spec.as_dict(), 'gpu': gpu,
        'protocol': 'from-scratch stability pilot, plasticity on throughout; chunk timings include final counter synchronization; no equilibrium claim',
        'paper': 'https://doi.org/10.1162/neco.2007.19.6.1437',
        'changed_from_paper': {'delay_ms': [1.5, 0.0]},
        'stopping_rules': {'population_rate_hz': args.abort_rate_hz,
            'sampled_weight_max_pa': 1000, 'nonfinite_neuron_state': 'abort'}}
    import pygenn
    record['genn_version'] = pygenn.__version__
    write_json(args.output / 'manifest.json', record)
    started = time.perf_counter()
    network = None
    rows = []
    status, error = 'complete', None
    try:
        network = GeNNBrunel(spec=spec, seed=args.seed, state_seed=args.seed,
            backend='cuda', build_path=args.output / 'build', recording_steps=0,
            precision='float', stdp_timing='arrival', stdp_tie_order='nest_causal_boundary',
            timing_enabled=False, record_spikes=False, collect_connectivity_stats=False)
        build_seconds = time.perf_counter() - started
        write_json(args.output / 'allocated_gpu_memory.json', command([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv']))
        previous = network.population_spike_counts()
        for step in range(0, round(args.sim_ms / DT_MS), round(args.chunk_ms / DT_MS)):
            ticks = min(round(args.chunk_ms / DT_MS), round(args.sim_ms / DT_MS) - step)
            wall = network.step(ticks)
            current = network.population_spike_counts()
            e = current[0] - previous[0]
            i = current[1] - previous[1]
            row = {'end_ms': (step + ticks) * DT_MS, 'ticks': ticks,
                   'wall_seconds': wall, 'us_per_step': wall * 1e6 / ticks,
                   'e_spikes': int(e.sum()), 'i_spikes': int(i.sum()),
                   'e_hz': float(e.mean() * 1000 / (ticks * DT_MS)),
                   'i_hz': float(i.mean() * 1000 / (ticks * DT_MS)),
                   'e_active': int(np.count_nonzero(e)), 'state': inspect_state(network, spec)}
            rows.append(row)
            print(row, flush=True)
            previous = current
            if max(row['e_hz'], row['i_hz']) >= args.abort_rate_hz:
                raise RuntimeError('Population rate exceeded pilot cutoff')
        weights = sample_weights(network, spec)
        if not any(row['e_spikes'] for row in rows):
            raise RuntimeError('Excitatory population remained silent')
    except Exception as exc:
        status, error = 'failed', repr(exc)
        raise
    finally:
        write_json(args.output / 'result.json', {'status': status, 'error': error,
            'build_seconds': locals().get('build_seconds'), 'chunks': rows,
            'weights': locals().get('weights'), 'process_wall_seconds': time.perf_counter() - started})
        if network is not None:
            network.close()


if __name__ == '__main__':
    main()
