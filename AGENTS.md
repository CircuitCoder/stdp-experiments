# Repository Guidance

This repository contains a simple STDP implementation that is being aligned with the reference implementation in `ref/`. Previous experiment logs and reports are in `copilot/`. The committed historical baseline used a current-based neuron model while the reference is conductance-based; ongoing work may already change that distinction, so verify the live source and worktree diff before relying on it.

Future work has two main, related tracks:

1. Experiment with the STDP algorithm and network parameters, recording enough context and results to make comparisons reproducible.
2. Study the reference implementation and align or reimplement the STDP behavior so that its mechanics and remaining differences are understood.

## Working Rules

1. Never use Git to commit code or alter repository history. The user will review and commit changes manually. Read-only Git commands may be used to inspect status, diffs, logs, blame, and repository history.
2. Never modify code without explicit confirmation from the user. For every task, first inspect the relevant context, form a concrete plan, and ask the user to confirm or amend that plan. Make code changes only after confirmation.
3. Ask questions and raise objections when appropriate. If anything is uncertain, ambiguous, inconsistent, or technically dubious, ask the user instead of over-analyzing it or agreeing merely to please them. The user's instructions and answers may contain mistakes; surface concerns directly and constructively.
4. Work takes place in an isolated NixOS container. Installing software and otherwise configuring or using the environment is allowed. Take particular care with deletion: do not delete files from this repository or its sibling directories without explicit user approval. If additional permissions would help, ask the user for them.

## Start Every Session Here

1. Read this file, then inspect `git status`, `git diff`, and the relevant current source before forming a plan. The worktree often contains substantial experiments that are newer than both `HEAD` and the reports in `copilot/`.
2. Treat current source plus the worktree diff as the implementation source of truth. Treat `copilot/*.md` and `copilot/tmp/*` as dated experimental evidence, not necessarily descriptions of the current code.
3. State which implementation state, dataset, command line, and evaluation protocol a result belongs to. Do not compare accuracy figures unless those conditions are compatible.
4. Keep investigation read-only until the user confirms the plan. After confirmation, keep edits tightly scoped and preserve unrelated user changes and experimental artifacts.

## Repository Map

- `src/main.rs`: CLI, dataset loading, presentation/retry/rest scheduling, training, class assignment, evaluation, checkpoint orchestration, and experiment diagnostics.
- `src/snn/network.rs`: network topology, active feedforward STDP updates, normalization and slow scaling, conductance routing, delays, runtime snapshots, checkpoint conversion, and Diehl-Cook defaults.
- `src/snn/neurons.rs`: the LIF neuron dynamics, conductance state, reversal potentials, refractory behavior, and adaptive threshold.
- `src/snn/weight.rs`: dense and one-to-one weight storage and traversal. Dense feedforward matrices are row-major `[pre * num_post + post]`.
- `src/snn/synapse.rs`: an older standalone pair-based `STDPSynapse`. Do not assume this is the live learning rule; the active feedforward update currently lives in `Network::tick` in `src/snn/network.rs`.
- `src/checkpoint.rs`: versioned binary training checkpoints and exact run-configuration capture.
- `src/xoshiro256pp.rs`: deterministic shuffling of the training order.
- `network.md`: zero-delay midpoint experiment summary, measured results, artifact layout, and guidance for deriving new network variants.
- `ref/`: a Git submodule containing the original Python 2/Brian 1 Diehl-Cook demo, pretrained weights, random recurrent connections, and cached activity. Its generated data, symlinks, and activity files may appear as untracked submodule contents; preserve them.
- `copilot/`: historical reports and experiment notes. `copilot/tmp/` contains commands, helper scripts, logs, and checkpoints from earlier sessions.
- `data/`: the root IDX files are Fashion-MNIST; `data/mnist/` contains real MNIST.

## Current Implementation Snapshot

Always verify this section against the live diff because it can become stale. When this guidance was expanded, committed `HEAD` still represented the earlier current-based model, while the uncommitted worktree contained a large conductance-based alignment rewrite.

The live worktree at that point included:

- conductance-based excitatory and inhibitory LIF populations with `ge` and `gi`, reversal potentials, exponential conductance decay, and separate E/I neuron constants;
- `784 -> E` feedforward connections, one-to-one `E -> I`, and dense `I -> E` without self-connections;
- random per-feedforward-synapse delays of up to 20 ticks;
- an elapsed-time representation of the reference pre/post traces, including the slow `post2before` gate for triplet potentiation;
- feedforward column normalization before each training presentation rather than after every STDP event;
- 700 presentation ticks and 300 zero-input rest ticks by default, corresponding to 0.5 ms per tick;
- runtime snapshots around evaluation so verification should not permanently alter the training state; and
- checkpoint format version 6. Older checkpoint versions are intentionally rejected, and resume requires the stored run configuration to equal the current CLI configuration.

Do not quote historical current-based results as measurements of this conductance-based worktree until it has been benchmarked directly.

## Reference Model Invariants

Use `ref/Diehl&Cook_spiking_MNIST.py` as the primary specification and verify details in the source rather than relying only on summaries. Important reference properties are:

- Python 2 with Brian 1, global simulation step `dt = 0.5 ms`;
- 784 Poisson inputs, 400 excitatory neurons, and 400 inhibitory neurons;
- dense input-to-excitatory connectivity, one-to-one excitatory-to-inhibitory connectivity, and inhibitory-to-all-other-excitatory connectivity;
- a 350 ms stimulus followed by 150 ms rest, or 700 and 300 Rust ticks respectively;
- conductance decay constants of 1 ms for excitation and 2 ms for inhibition;
- feedforward delay sampled from the 0-10 ms interval;
- normalization of every input-to-excitatory column to a sum of 78 before each accepted training presentation attempt;
- retry with increased input intensity when the excitatory population emits fewer than five spikes; neural state is allowed to evolve through the intervening rest rather than being reset;
- adaptive threshold only on excitatory neurons during training; and
- trace-based triplet STDP: depression uses the `post1` trace, while potentiation uses `pre * post2before`, with weights clamped to `[0, 1]`.

When aligning behavior, investigate event ordering explicitly. In particular, check whether delayed presynaptic arrival or source emission drives STDP, when traces are sampled and reset on same-tick events, whether conductances and theta freeze during refractory periods, and whether the Rust integration/decay ordering matches Brian 1. Small deterministic trace tests are preferable to inferring these details from final accuracy.

## Datasets And Evaluation

The CLI defaults to Fashion-MNIST and `data/`. A real-MNIST experiment must include:

```text
--dataset mnist --data-path data/mnist
```

Training labels are not used by STDP. Marking assigns each excitatory neuron to the class for which it had the highest mean spike count, and testing predicts from class-normalized population spike totals. Marking can use the validation or test split; multiple disjoint marking sets can be evaluated at each checkpoint.

The cached reference activity reproduces 91.56% under the stock demo evaluator. A prior local train-from-scratch reference run reported 89.04%. Both are simple-demo results in which the same test-mode activity is used for neuron assignment and scoring, which is optimistic and differs from the paper's stricter protocol. The paper derives assignments from a separate training-set inference pass. Always label which protocol is being used.

Historical Rust results in `copilot/report.md`, `copilot/improvement*.md`, `copilot/report.diff.md`, `copilot/param.md`, `copilot/fix.diff.md`, and `copilot/aligned-current-based.md` cover different and often older implementations. Useful conclusions include the importance of the explicit rest phase, correct time/weight scaling, E/I parameter separation, and evaluation protocol. Their absolute accuracy values are not a current regression baseline unless the exact code and command are reconstructed.

## Experiment Practice

1. Run unit tests before and after implementation changes. Add deterministic micro-tests for event timing, traces, refractory behavior, delay delivery, normalization, and checkpoint restoration when those areas change.
2. Use release builds for performance measurements. Debug-mode throughput is not representative.
3. Start with a small pilot that can expose silence, runaway firing, retry storms, class collapse, and invalid weight/threshold ranges before starting a long run. A reference-sized run can take many hours or days in this environment.
4. Record the complete command, source/worktree state, wall time, dataset, split sizes, output-neuron count, seed, accuracy, firing statistics, class-assignment distribution, weight statistics, and log path. Put disposable run outputs under `copilot/tmp/` unless the user chooses another location.
5. Change one mechanism or a deliberately small parameter group at a time. Keep a control run so an improvement can be attributed to the change.
6. Do not judge alignment by accuracy alone. Also compare spike counts, retry frequency, per-neuron firing distribution, theta distribution, conductance ranges, feedforward column sums, weight saturation/sparsity, and class-assignment balance.
7. Evaluation runs should have plasticity disabled and should restore all mutable runtime state afterward. Verify this whenever new neuron, synapse, delay, RNG, or homeostasis state is introduced.

## Reference Experiment Isolation

Brian 1 reference runs use relative paths and fixed filenames. Treat the working
directory as part of the experiment configuration.

1. Give every training case its own runtime directory containing its copied
   source and separate `random/`, `weights/`, `activity/`, and `results/`
   directories. Give every checkpoint evaluation another fresh runtime
   directory. Never train and evaluate in the source template directory.
2. Use a stable training case name and a separate evaluation tag in every
   directory, log, service, and score record. This is especially important when
   evaluating one checkpoint with several inference-only parameter settings.
3. Launchers should fail if their output directory or log already exists. Do not
   silently reuse a directory: Brian's unsuffixed final weights and numbered
   checkpoints can otherwise be mistaken for another run or overwritten.
4. Copy the source file into each runtime directory instead of generating a
   derivative source at launch time. Record the source hash and explicit
   environment variables that define the case. A source generator adds another
   implementation whose output and quoting must be audited.
5. Use the pinned Python 2/Brian 1 environment under `ref/env` for reference
   work. Preserve the launcher's `PATH`, `LD_LIBRARY_PATH`, compiler flags, and
   Python dependency versions; an interactive shell without those variables can
   fail even when the background service is healthy.
6. Long jobs should run as named background services with stdout and stderr in
   unique files under `ref/logs/`. Record the complete service command and
   environment. Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when running
   independent cases concurrently so library thread pools do not obscure the
   intended process-level parallelism.

### Checkpoint Integrity

- A reference checkpoint is an immutable matched pair such as
  `XeAe30000.npy` and `theta_A30000.npy`. Wait until both files have stable size
  and modification time, then copy and hash them together with the exact source
  and recurrent matrices before evaluation.
- Keep the training case separate from the evaluation tag. The former selects
  the checkpoint; the latter identifies the inference configuration and output.
  This prevents inference sweeps from colliding or being misreported as
  independently trained networks.
- Numbered and unsuffixed files have different meanings. Never infer the sample
  count of `XeAe.npy` from its directory name; require a completion marker and
  artifact manifest before treating it as a final checkpoint.
- Brian reference checkpoints currently contain weights and adaptive thresholds,
  not membrane voltages, conductances, refractory timers, delay queues, STDP
  traces, Poisson/RNG state, or presentation retry state. They are sufficient for
  a fresh plasticity-off inference pass, but restarting training from them is not
  an exact continuation. Label any such run as a branched resume and record the
  reset-state caveat.
- Checkpoint evaluation must use copied artifacts and plasticity-off mode. It
  must not mutate the training directory or compete for the same activity/result
  filenames.

### Protocol Labels

Do not combine these accuracy measurements in one curve without an explicit
conversion or caveat:

- the order-dependent accuracy over the preceding training block;
- a 1,000-image checkpoint probe;
- the stock 10,000-image simple-demo evaluation; and
- the paper protocol with assignments derived from a separate training-set pass.

The first three local evaluators may assign neurons and score predictions from
the same activity, which is optimistic. Record the inference-time inhibition and
other runtime parameters as well as the training parameters. An inference-only
parameter sweep can diagnose readout suppression, but it is not evidence that
several distinct networks were trained.

## Numerical Stability Experiments

Reducing synaptic delay can align many conductance arrivals within one global
tick. Conductance-based dynamics do not by themselves make a fully explicit
voltage update stable: at `dt = 0.5 ms`, a sufficiently large total conductance
can make the explicit step overshoot reversal-potential bounds and then diverge.

- Add a deterministic one-step verifier whenever the integration rule changes.
  Compare the generated Brian state updater against an independently calculated
  step, including conductance decay and refractory behavior.
- Run a short smoke test and a small accuracy pilot before a long training run.
  Abort on non-finite state, implausible voltage/conductance ranges, a configured
  per-stimulus spike limit, retry storms, or weight/threshold explosion.
- The successful zero-delay reference variant in `ref/zero_delay_midpoint_v1/`
  uses exact exponential conductance decay and an exact constant-conductance
  voltage step evaluated at analytically decayed midpoint conductances. This is a
  stable frozen-conductance approximation, not the full coupled closed-form
  solution. See `network.md` and the experiment manifest before deriving from it.
- Keep integration changes separate from topology, learning-rule, and parameter
  changes. Accuracy alone cannot distinguish a numerical fix from a changed
  model.

## Derived Network Workflow

1. Select a documented baseline and copy its source into a new immutable
   experiment-family template. Do not edit the template or runtime directory of
   an active run.
2. Create one clean runtime directory per case. Start structural, learning-rule,
   and training-parameter derivatives from the same recorded random feedforward
   and recurrent artifacts unless intentionally studying initialization.
   Inference-only derivatives may reuse a learned checkpoint, but must receive
   distinct evaluation tags.
3. Write a manifest before launch: parent family, source and artifact hashes,
   exact changed variables, dataset/order, simulator/dependency versions,
   training and inference settings, checkpoint interval, evaluation protocol,
   stopping rules, and log/service names.
4. Change one mechanism or a small deliberate parameter group. Use paired
   controls and, because not all randomness is owned, multiple repetitions when
   the expected effect is small.
5. Use staged budgets. A short pilot should reject silence, runaway activity,
   and obvious collapse. Roughly 30,000 accepted samples is useful for screening,
   but it is only half of one 60,000-image MNIST pass and is not proof of final
   convergence. Run promising cases through at least 60,000 samples and use the
   full 180,000 when making a reference-equivalent final claim.
6. Evaluate immutable checkpoints at regular intervals with the same sample set
   and protocol. Require several stable accuracy points plus firing, retry,
   threshold, weight, conductance, and class-balance diagnostics before calling a
   model saturated.
7. Preserve failed cases and their manifests/logs. A stopped collapse or
   divergence is useful comparative evidence when its checkpoint identity and
   protocol remain auditable.

## Reproducibility Caveat

`--rng-seed` currently controls training-order shuffling through `Xoshiro256PlusPlus`, but it does not fully determine a run. Feedforward initialization, feedforward delays, Poisson spikes, and optional noise use `rand::random`. Identical CLI commands can therefore produce different results. Do not claim exact reproducibility from `--rng-seed` alone, and prefer multiple runs or reported ranges when comparing parameter settings. Full RNG ownership and checkpointed RNG state remain alignment and reproducibility work items.

## Build And Validation

Normal commands are:

```sh
cargo test
cargo test --release
cargo run --release -- --help
cargo run --release -- <experiment arguments>
```

The repository's `.cargo/config.toml` enables `-C target-cpu=native`, so compiled artifacts are machine-specific. In the managed container, `/root/.cargo` may be readable but not writable. If Cargo fails while unpacking a cached dependency, create a writable temporary copy and use it offline:

```sh
cp -a /root/.cargo /tmp/stdp-cargo-home
CARGO_HOME=/tmp/stdp-cargo-home cargo test --offline
```

Do not repeat the copy over an existing destination because that can create an unintended nested directory; choose a fresh temporary path instead. At the time this guidance was expanded, the worktree test suite reported 28 passed tests, 1 ignored test, and no failures, with several unused-code warnings. Re-run the suite rather than treating that count as permanent.
