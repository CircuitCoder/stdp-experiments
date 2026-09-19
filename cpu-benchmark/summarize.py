#!/usr/bin/env python3
"""Report complete single-network CPU latency runs; reject replica throughput."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def latency_samples(measurement, batches):
    if measurement.get('metric') != 'single_network_latency' or measurement.get('replicas') != 1:
        raise ValueError('Expected single-network latency; independent-replica throughput is not comparable')
    if len(batches) != measurement['repetitions'] or not batches:
        raise ValueError('Incomplete timing repetitions')
    samples = []
    for batch in batches:
        if (batch.get('metric') != 'single_network_latency' or batch['replicas'] != 1
                or batch['simulation_threads_per_replica'] != 1 or len(batch['workers']) != 1):
            raise ValueError('Expected one single-threaded network in every repetition')
        worker = batch['workers'][0]
        if len(worker['cpu_affinity']) != 1:
            raise ValueError('Expected the simulation to be pinned to one logical CPU')
        expected = worker['wall_seconds'] * 1e6 / worker['simulation_steps']
        if not all(math.isclose(v, expected, rel_tol=1e-12) for v in [worker['us_per_step'], batch['us_per_step']]):
            raise ValueError('Latency must use the network wall timer and its own step count')
        samples.append(expected)
    if not math.isclose(statistics.median(samples), measurement['median_us_per_step'], rel_tol=1e-12):
        raise ValueError('Summary disagrees with raw single-network timings')
    return samples


def write_csv(path, rows):
    if rows:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genn', type=Path, required=True)
    parser.add_argument('--nest', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.genn / 'summary.json').read_text())
    manifest = json.loads((args.genn / 'manifest.json').read_text())
    gpu = json.loads((ROOT / 'genn-sweep/baseline-20260907-matrix.json').read_text())
    if manifest['schema'] != 'genn-cpu-latency-v2':
        raise ValueError('Use a new single-network latency run; historical replica runs are not accepted')
    if set(summary) != set(gpu['machines']['RTX3090']['cases']):
        raise ValueError('This report requires the complete 23-case matrix')
    rows, diagnostics = [], []
    for case, measurement in summary.items():
        batches = [json.loads(p.read_text()) for p in sorted((args.genn / 'timing' / case).glob('*/batch.json'))]
        samples = latency_samples(measurement, batches)
        rows.append({'case': case, 'networks': 1, 'simulation_threads': 1,
            'cpu_median_us_per_step': statistics.median(samples),
            'cpu_min_us_per_step': min(samples), 'cpu_max_us_per_step': max(samples),
            'cpu_iqr_us_per_step': measurement['iqr'],
            **{f'{machine.lower()}_us_per_step': data['cases'][case]['median_us_per_step']
               for machine, data in gpu['machines'].items()}})
        control = measurement['diagnostic']
        checks, events = control['diagnostic'], control['event_counters']
        reference = (gpu['machines']['RTX3090']['cases'][case]['diagnostic']['event_counters']
                     if case.startswith('mnist_') else
                     gpu['cross_machine_dynamics'][case]['population_totals']['RTX3090'])
        diagnostics.append({'case': case, 'simulation_steps': control['simulation_steps'],
            'attempts': control.get('attempts'), 'cpu_exc_spikes': events['excitatory_spikes'],
            'cpu_inh_spikes': events['inhibitory_spikes'], 'gpu_exc_spikes': reference['excitatory_spikes'],
            'gpu_inh_spikes': reference['inhibitory_spikes'],
            'weight_min': checks['weight_min'], 'weight_max': checks['weight_max'],
            'weight_mean': checks['weight_mean'], 'theta_min_mv': checks.get('theta_min_mv'),
            'theta_max_mv': checks.get('theta_max_mv'), 'column_sum_min': checks.get('column_sum_min'),
            'column_sum_max': checks.get('column_sum_max'),
            'population_fano_3ms': checks.get('population_fano_3ms')})
    nest_results = [{'path': str(p), 'summary': json.loads((p / 'summary.json').read_text())} for p in args.nest]
    nest_rows = []
    for entry in nest_results:
        r = entry['summary']
        nest_rows.append({'rule': r['rule'], 'threads': r['threads'], 'networks': 1,
            'median_us_per_step': r['median_us_per_step'], 'min_us_per_step': r['range'][0],
            'max_us_per_step': r['range'][1], 'exc_rate_hz': r['diagnostic']['exc_rate_hz'],
            'inh_rate_hz': r['diagnostic']['inh_rate_hz'], 'raw_path': entry['path']})
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'timings.csv', rows)
    write_csv(args.output / 'diagnostics.csv', diagnostics)
    write_csv(args.output / 'nest-timings.csv', nest_rows)
    result = {'genn_path': str(args.genn), 'genn_manifest': manifest, 'genn': rows,
        'nest': nest_results, 'scope': 'single-network elapsed time per timestep for CPU, NEST and GPU'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    config = manifest['configuration']
    lines = ['# GeNN CPU single-network latency, 2026-09-10', '',
        f'All 23 cases completed with one network, one simulation thread, and {config["repetitions"]} '
        f'timed repetitions per case. The Ryzen 9 7950X guest affinity was {manifest["allowed_cpus"]}. '
        'Other host jobs continued running; this is a contended-host measurement, not an isolated peak.', '',
        '**Correction:** the earlier 32-replica aggregate figures measure independent-network throughput. '
        'They do not measure timestep latency and cannot support a CPU speedup over a single GPU network. '
        'This report uses only the elapsed timer of one network divided by that network\'s executed steps. '
        'Repetitions run sequentially. No division by a replica or core count is performed.', '',
        '## GeNN parallelization assessment', '',
        'The installed GeNN 5.4 CPU backend executes one simulation thread. The upstream experimental '
        'ISPC backend and PR #710 were also inspected: their generated foreach loops use SIMD, but no task '
        'launch, OpenMP region or CPU thread pool distributes this workload across cores. '
        'No usable multicore backend was enabled. We used the requested single-core fallback. '
        'This was a backend capability investigation, not a completed implementation of a new parallel backend.', '',
        'A new backend would need explicit ownership of shared RNG state, spike lists and synaptic current '
        'accumulators, plus barriers that preserve synapse/neuron and STDP ordering. Its complete timestep '
        'timer would have to include those barriers. Merely enabling a compiler OpenMP flag does not provide this.', '',
        '[ISPC development report](https://genn-team.github.io/posts/developing-an-ispc-backend-for-genn-bridging-gpu-and-cpu-performance-for-neural-network-simulations.html); '
        '[open ISPC pull request](https://github.com/genn-team/genn/pull/710). '
        'Audited upstream ISPC revision: 8aef8f129b5847631cd104b40bc85786efd66b93; '
        'PR head: a70e7382a777dcde40352ebee24b5b3cca90b270. '
        'Source copies and hashes: copilot/tmp/genn_cpu_parallel_audit_20260910_a/.', '',
        '## Elapsed microseconds per timestep', '',
        'CPU values are the median and full range of five sequential runs. GPU columns are the existing '
        'FP32 single-network baseline. CPU and GPU use the same model definitions and input artifacts but '
        'different random generators; diagnostic activity is retained separately and is not bitwise identical.', '',
        '| Case | CPU, 1 thread | CPU range | RTX 3090 | A100 | A800 | H800 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {r["cpu_median_us_per_step"]:.3f} '
            f'| {r["cpu_min_us_per_step"]:.3f}–{r["cpu_max_us_per_step"]:.3f} '
            f'| {r["rtx3090_us_per_step"]:.3f} | {r["a100_us_per_step"]:.3f} '
            f'| {r["a800_us_per_step"]:.3f} | {r["h800_us_per_step"]:.3f} |')
    if nest_rows:
        lines += ['', '## NEST: one network with real internal threading', '',
            'These earlier runs already measure single-network latency, including NEST synchronization. '
            'They use the custom current-default Brunel port. The NEST engine uses -O3; its initial Nix '
            'build stripped -march=native, while the custom module retained it. Neuron/plastic state is FP32 '
            'and event-buffer sums are FP64. Native NEST RNG and connectivity differ from GeNN and depend '
            'on thread count. Background contention changed during the matrix, so these are not clean SMT scaling results.', '',
            '| Rule | Threads | Median µs/step | Range |', '|---|---:|---:|---:|']
        for r in nest_rows:
            lines.append(f'| {r["rule"]} | {r["threads"]} | {r["median_us_per_step"]:.3f} '
                f'| {r["min_us_per_step"]:.3f}–{r["max_us_per_step"]:.3f} |')
    lines += ['', '## Protocol and validation', '',
        f'GeNN/PyGeNN {manifest["genn"]}, GCC optimized with -O3 -march=native and '
        'NIX_ENFORCE_NO_NATIVE=0; no optional GeNN fast-math. Simulation state is FP32 and host '
        'normalization FP64. BLAS and OpenMP environment thread counts are one. Affinity pins the process '
        'to one guest logical CPU; the host SMT sibling is not reserved or disabled.', '',
        f'Real MNIST uses data/mnist and the frozen baseline_inputs_20260907_v2 bundle. Each repetition '
        f'restores the 10,000-image weights/theta checkpoint with other runtime state reset, then trains '
        f'{config["samples"]} accepted images from index 10000, seed {config["seed"]}. '
        'This is branched continuation. The timed region includes normalization, the host control loop, '
        'all retries, 700 stimulus ticks and 300 rest ticks per attempt. The denominator includes every executed tick. '
        'No accuracy or convergence comparison is made.', '',
        'Brunel uses 9000 E + 2250 I neurons, fixed indegrees 450/112 with replacement and without '
        'recurrent autapses, 6,322,500 connections, dt=0.1 ms, next-tick delivery, arrival-timed '
        'causal-boundary STDP, sqrt(20) delivery scaling and external-rate scales 0.47/0.32. '
        'The timer covers 1000 ms / 10000 steps after 100 ms presimulation.', '',
        'Construction, compilation, loading, presimulation and result serialization are excluded, matching '
        'the GPU baseline boundaries. Parent process startup/rendezvous is excluded; this is harness setup, '
        'not simulation synchronization. The GeNN CPU step function completes synchronously, so no unmeasured '
        'device work remains after its loop returns.', '',
        'Every timing repetition exactly reproduced its CPU diagnostic step count, total spikes and event '
        'counters. Diagnostics guard finite state, firing, retries, weights, thresholds and normalization. '
        'The report generator rejects replica runs, incomplete repetitions, unpinned workers and mismatched '
        'time/step denominators. The deterministic NEST checks cover current integration, refractory '
        'behavior, next-step delivery and simultaneous STDP events.', '',
        'Power estimation is deferred. This WSL2 container exposes no readable package energy counter; '
        'the raw energy records are null. No per-watt claim or power estimate is made.', '',
        '## Artifacts', '',
        f'- Complete raw run, source copies, command and hashes: `{args.genn}`',
        f'- Input manifest SHA-256: `{manifest["input_manifest_sha256"]}`',
        '- GPU baseline: `genn-sweep/baseline-20260907-matrix.json`',
        '- [Full latency table](timings.csv)', '- [Activity and state diagnostics](diagnostics.csv)',
        '- [Machine-readable report and provenance](summary.json)']
    if nest_rows:
        lines.append('- [NEST timings](nest-timings.csv)')
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
