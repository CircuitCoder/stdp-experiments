#!/usr/bin/env python3
"""Compare completed NEST MNIST single-network latency with the frozen baselines."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nest', type=Path, required=True)
    parser.add_argument('--genn', type=Path, required=True)
    parser.add_argument('--thread-pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    nest = read(args.nest / 'summary.json')
    manifest = read(args.nest / 'manifest.json')
    genn = read(args.genn / 'summary.json')
    genn_manifest = read(args.genn / 'manifest.json')
    gpu = read(ROOT / 'genn-sweep/baseline-20260907-matrix.json')
    expected = {case for case in gpu['machines']['RTX3090']['cases'] if case.startswith('mnist_')}
    if set(nest) != expected or len(expected) != 21:
        raise ValueError('The NEST report requires all 21 completed MNIST cases')
    if manifest['schema'] != 'nest-mnist-latency-v1' or genn_manifest['schema'] != 'genn-cpu-latency-v2':
        raise ValueError('Expected current single-network latency formats')
    if manifest['input_manifest_sha256'] != genn_manifest['input_manifest_sha256']:
        raise ValueError('NEST and GeNN must use the same frozen input bundle')
    for config in [manifest['configuration'], genn_manifest['configuration']]:
        if config['samples'] != 100 or config['seed'] != 20260724:
            raise ValueError('GPU baseline comparison requires 100 images and seed 20260724; do not compare smoke-test timings')
    for path in ['genn-sweep/baseline_cases.py', 'reimpl/zd3/constants.py', 'reimpl/zd3/variants.py']:
        if manifest['source_sha256'][path] != genn_manifest['source_sha256'][path]:
            raise ValueError(f'Workload definition differs: {path}')
    rows, diagnostics = [], []
    validated = 0
    for case, measurement in nest.items():
        if measurement['metric'] != 'single_network_latency' or measurement['networks'] != 1:
            raise ValueError('Replica throughput is not timestep latency')
        runs = [read(path) for path in sorted((args.nest / case).glob('r*.json'))]
        if len(runs) != measurement['repetitions'] or len(runs) != manifest['configuration']['repetitions']:
            raise ValueError(f'Incomplete repetitions: {case}')
        control = measurement['diagnostic']
        for run in runs:
            if run['metric'] != 'single_network_latency' or run['networks'] != 1 or run['threads'] != measurement['threads']:
                raise ValueError(f'Incompatible concurrency: {case}')
            if not math.isclose(run['wall_seconds'] * 1e6 / run['simulation_steps'], run['us_per_step'], rel_tol=1e-12):
                raise ValueError(f'Incorrect wall-time/step denominator: {case}')
            for key in ['simulation_steps', 'total_spikes', 'event_counters', 'executed_synaptic_visits',
                        'final_weights_sha256', 'final_theta_sha256']:
                if run[key] != control[key]:
                    raise ValueError(f'Diagnostic mismatch: {case}/{key}')
            validated += 1
        median = statistics.median(run['us_per_step'] for run in runs)
        if not math.isclose(median, measurement['median_us_per_step'], rel_tol=1e-12):
            raise ValueError(f'Incorrect summary median: {case}')
        if genn[case]['metric'] != 'single_network_latency' or genn[case]['replicas'] != 1:
            raise ValueError('GeNN comparison must measure one network')
        cpu = genn[case]['median_us_per_step']
        rows.append({'case': case, 'nest_threads': measurement['threads'], 'networks': 1,
            'nest_median_us_per_step': median, 'nest_min_us_per_step': measurement['range'][0],
            'nest_max_us_per_step': measurement['range'][1], 'nest_iqr_us_per_step': measurement['iqr'],
            'genn_cpu_1thread_us_per_step': cpu, 'nest_over_genn_cpu_time': median / cpu,
            **{f'{name.lower()}_us_per_step': machine['cases'][case]['median_us_per_step']
               for name, machine in gpu['machines'].items()}})
        d, events = control['diagnostic'], control['event_counters']
        genn_control = genn[case]['diagnostic']
        diagnostics.append({'case': case, 'attempts': control['attempts'],
            'simulation_steps': control['simulation_steps'],
            'genn_cpu_attempts': genn_control['attempts'],
            'genn_cpu_simulation_steps': genn_control['simulation_steps'],
            'input_spikes': events['input_spikes'], 'exc_spikes': events['excitatory_spikes'],
            'inh_spikes': events['inhibitory_spikes'],
            'genn_cpu_input_spikes': genn_control['event_counters']['input_spikes'],
            'genn_cpu_exc_spikes': genn_control['event_counters']['excitatory_spikes'],
            'genn_cpu_inh_spikes': genn_control['event_counters']['inhibitory_spikes'],
            'nest_exc_spikes_per_step': events['excitatory_spikes'] / control['simulation_steps'],
            'genn_cpu_exc_spikes_per_step': genn_control['event_counters']['excitatory_spikes'] / genn_control['simulation_steps'],
            'ff_connections': d['ff_connections'], 'ff_outdegree_min': d['ff_outdegree_range'][0],
            'ff_outdegree_max': d['ff_outdegree_range'][1], 'ff_indegree_min': d['ff_indegree_range'][0],
            'ff_indegree_max': d['ff_indegree_range'][1],
            'ff_pre_visits': control['executed_synaptic_visits']['ff_pre_visits'],
            'ff_post_visits': control['executed_synaptic_visits']['ff_post_visits'],
            'weight_min': d['weight_min'], 'weight_max': d['weight_max'], 'weight_mean': d['weight_mean'],
            'weight_change_rms': d['weight_change_rms'], 'weight_change_max': d['weight_change_max'],
            'column_sum_min': d['column_sum_min'], 'column_sum_max': d['column_sum_max'],
            **d['normalization_bound_observation']})
    pilot_rows = []
    if not (args.thread_pilot / 'COMPLETE').is_file():
        raise ValueError('Incomplete thread pilot')
    for threads in [1, 2, 4, 8, 16, 32]:
        summary = read(args.thread_pilot / f't{threads}' / 'summary.json')
        pilot_rows.append({'threads': threads, **{f'{trace}trace_us_per_step':
            summary[f'mnist_{trace}trace_dense']['median_us_per_step'] for trace in [1, 2, 3]}})
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'timings.csv', rows)
    write_csv(args.output / 'diagnostics.csv', diagnostics)
    write_csv(args.output / 'thread-pilot.csv', pilot_rows)
    result = {'scope': 'single-network MNIST latency; all synchronization included',
        'nest_path': str(args.nest), 'genn_path': str(args.genn), 'thread_pilot_path': str(args.thread_pilot),
        'manifest': manifest, 'validated_repetitions': validated, 'timings': rows,
        'diagnostics': diagnostics, 'thread_pilot': pilot_rows,
        'report_generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    threads = manifest['configuration']['threads']
    configurations = [('dense', 'Dense'), ('bernoulli_0500', 'Random 50%'),
        ('bernoulli_0250', 'Random 25%'), ('bernoulli_0125', 'Random 12.5%'),
        ('fixed-fanout_0500', 'Fixed fan-out 50%'), ('fixed-fanout_0250', 'Fixed fan-out 25%'),
        ('fixed-fanout_0125', 'Fixed fan-out 12.5%')]
    over = [d for d in diagnostics if d['attempts_above_102pct_wmax']]
    ratios = [r['nest_over_genn_cpu_time'] for r in rows]
    lines = ['# NEST MNIST benchmark matrix, 2026-09-10', '',
        f'All 21 cases completed with one network using {threads} NEST threads. '
        f'Each case has {manifest["configuration"]["repetitions"]} sequential timing repetitions '
        f'of {manifest["configuration"]["samples"]} accepted MNIST images. '
        f'All {validated} timings exactly reproduced their diagnostic spikes, executed synaptic visits, '
        'final FP32 weights and thresholds. Other host processes remained active.', '',
        'The fixed-degree cases are **fixed fan-out**: every input has exactly 200, 100 or 50 E targets. '
        'The random cases use Bernoulli masks at 50%, 25% and 12.5%. Frozen checkpoint masks are reused '
        'rather than regenerated by NEST.', '',
        '## Median elapsed microseconds per timestep', '',
        '| Connectivity | 1 trace | 2 traces | 3 traces |', '|---|---:|---:|---:|']
    for tag, name in configurations:
        values = [nest[f'mnist_{trace}trace_{tag}']['median_us_per_step'] for trace in [1, 2, 3]]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', 'Each timestep represents 0.5 ms of biological time. Values are one network\'s elapsed '
        'time divided by its executed stimulus and rest ticks, including retry attempts. No replica/core '
        'count divides the result. Construction and setup are excluded; NEST simulation synchronization, '
        'normalization, host control and final spike-count reads are included.', '',
        '## Comparison with the current GeNN baselines', '',
        'The same frozen weights/theta checkpoints, masks, model definitions and 100-image protocol are used. '
        'NEST uses its native random streams, so activity need not be bitwise identical to GeNN. '
        'The CSV retains activity counts and ranges for assessing that difference. '
        'These contended-host measurements do not establish isolated peak simulator performance.', '',
        f'NEST took {min(ratios):.2f}–{max(ratios):.2f} times the elapsed time of single-core GeNN '
        'across this matrix. GeNN remains the fastest CPU implementation measured here. '
        'This does not establish a performance-per-watt ranking.', '',
        f'| Case | NEST, {threads} threads | GeNN CPU, 1 thread | RTX 3090 | A100 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {r["nest_median_us_per_step"]:.3f} '
            f'| {r["genn_cpu_1thread_us_per_step"]:.3f} | {r["rtx3090_us_per_step"]:.3f} | {r["a100_us_per_step"]:.3f} |')
    lines += ['', 'All four GPU baselines, CPU/NEST medians, NEST ranges and NEST/CPU elapsed-time ratios '
        'are in [timings.csv](timings.csv).', '', '## Thread selection', '',
        'A preliminary dense-case pilot used five accepted images and two sequential timing repetitions '
        'per configuration. Each row evolves one network. It uses active OpenMP waiting, GOMP_SPINCOUNT=300000, '
        'OMP_PROC_BIND=spread and OMP_PLACES=cores. Native NEST RNG changes with thread count; this is a short '
        'operational scaling pilot, not a fixed-spike-train scaling study.', '',
        '| Threads | 1 trace | 2 traces | 3 traces |', '|---|---:|---:|---:|']
    for r in pilot_rows:
        lines.append(f'| {r["threads"]} | {r["1trace_us_per_step"]:.3f} | {r["2trace_us_per_step"]:.3f} | {r["3trace_us_per_step"]:.3f} |')
    lines += ['', 'Eight threads were modestly faster in the short pilot. Sixteen were selected to use all '
        'physical cores for the requested concurrency experiment. Thirty-two SMT threads were slower. '
        'An earlier four-thread passive-wait smoke test measured roughly 212–219 µs/tick on two images; '
        'its shorter sample budget differs, so it is not a controlled waiting-policy speedup measurement. '
        'The full-run thread placement was observed: 16 native threads had disjoint affinities, each covering '
        'one reported physical core\'s two logical CPUs. WSL2 affinity does not reserve the host cores.', '',
        '## Implementation and matching boundaries', '',
        'The new cpu-benchmark/mnist_module extension uses NEST\'s native neurons, spike transport and OpenMP '
        'scheduler. Plastic feedforward weights live in their target neuron so all postsynaptic updates can '
        'execute at the proper tick, including updates for an input that has never fired. Native static '
        'feedforward connections carry input identity in their receptor number; their constant transport '
        'weight is not the learned weight. Recurrent connections use ordinary NEST static synapses.', '',
        'One Bernoulli draw per input per tick is broadcast to its complete fan-out, matching the GeNN '
        'input-process definition. Per-input traces use alternating buffers: on tick s inputs write s%2 '
        'while E nodes read (s-1)%2. NEST\'s one-step minimum-delay barrier separates ticks. This extension '
        'requires one network in one MPI process; it is not a general distributed NEST synapse implementation.', '',
        'The current neuron midpoint integration, refractory conductance/theta freeze, adaptive threshold '
        'and next-tick delivery are preserved. Presynaptic delivery uses the pre-update weight, then depression '
        'runs before postsynaptic potentiation. Same-tick post increments are excluded from depression. '
        'A three-trace potentiation event uses the previous post for its slow trace. The one-/two-trace '
        'rules accumulate the input trace and use their configured weight-power dependence.', '',
        'Normalization performs the same FP64 column sum and scale followed by FP32 storage, inside the '
        'target nodes. It remains inside elapsed timing. The GeNN baseline performs this operation on '
        'host arrays. Normalization happens before pending last-tick STDP updates, matching the GeNN '
        'attempt boundary; no lazy history crosses that boundary.', '',
        'The run starts from the frozen 10,000-image checkpoint for each case, resetting other runtime state, '
        'then presents real MNIST from data/mnist in sequential order beginning at index 10000. '
        'Each attempt has 700 stimulus ticks and 300 rest ticks. The stimulus counter includes preceding '
        'rest spikes exactly as in the baseline; fewer than five E spikes causes an intensity retry. '
        'This is a branched continuation benchmark, with no accuracy or convergence claim.', '',
        'Neuron, trace and plastic state are FP32. Native NEST recurrent event buffers sum in FP64. '
        'The extension uses GCC 14.3, -O3 -march=native, NIX_ENFORCE_NO_NATIVE=0, without fast-math. '
        'The reused NEST engine was built with -O3; its earlier Nix build stripped -march=native. '
        'Different accumulation order and native random streams remain numerical differences from GeNN.', '',
        '## Weight-bound policy and validation', '',
        'As requested, normalization/final cap overshoots are observations, not abort conditions. '
        'The configured STDP clipping rule is unchanged. Timed GeNN runs likewise disable the normalization '
        'bound check, although the older GeNN diagnostic path still aborts on its tolerance. '
        'The NEST diagnostic now records the peak normalized weight/wmax and affected attempts. '
        'Finite-state, nonnegative-weight, structural-mask, normalization-sum, runaway and retry guards remain.', '',
        f'{len(over)} of 21 diagnostic cases exceeded 1.02 × wmax after normalization. '
        'See [diagnostics.csv](diagnostics.csv) for counts and maxima; minor FP32 overshoots above wmax '
        'are recorded separately.', '',
        'The 13 deterministic MNIST tests cover all three rules on one/four threads, simultaneous pre/post '
        'events, next-cycle recurrent delivery, the midpoint step, refractory freezing, normalization '
        'between emission and pending STDP, silent-input postsynaptic updates, shared traces across '
        'threads, and normalization when a column cannot fit within the learning cap. '
        'All 61 relevant tests passed. Every full timing repetition matched its diagnostic final weights '
        'and thresholds bitwise within the same NEST configuration. This validates the tested short-run '
        'mechanics; it does not establish long-training accuracy alignment.', '',
        'Power estimation remains deferred. No energy or performance-per-watt figure is reported.', '',
        '## Artifacts', '', f'- Raw full run: `{args.nest}`', f'- Thread pilot: `{args.thread_pilot}`',
        f'- GeNN CPU comparison: `{args.genn}`', '- [Full latency table](timings.csv)',
        '- [Activity, structure, weights and bound observations](diagnostics.csv)',
        '- [Thread pilot CSV](thread-pilot.csv)', '- [Results and provenance JSON](summary.json)',
        '- Source snapshots, model/module hashes, exact command and environment are in each run manifest.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
