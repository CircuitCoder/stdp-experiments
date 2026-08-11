# STDP equilibrium regimes in the NEST Brunel benchmarks

## Summary

The equilibrium shape of a plastic synaptic-weight distribution is controlled
primarily by the weight dependence of the STDP rule. The neuron and network
dynamics still matter because they determine the pre/post spike correlations,
and can shift or destroy an equilibrium, but changing from an additive to a
weight-dependent synapse is the direct mechanism behind the usual bimodal
versus unimodal distinction.

| Plasticity rule | Weight dependence | Expected equilibrium |
|---|---|---|
| Additive STDP with hard bounds | Update size is independent of the current weight | Bimodal, with mass near zero and `Wmax` |
| Multiplicative or soft-bound STDP | Updates shrink near one or both bounds | Unimodal, generally interior and possibly skewed |
| Morrison power-law STDP | Potentiation is proportional to `w^mu`; depression is proportional to `w` | Unimodal and approximately Gaussian in the balanced random network |

This distinction is summarized in Figure 5 of Morrison, Diesmann, and
Gerstner's review [1]. The primary additive result is due to Song, Miller, and
Abbott [2], while van Rossum, Bi, and Turrigiano demonstrated stable Hebbian
learning with a unimodal, positively skewed distribution [3]. Morrison,
Aertsen, and Diesmann introduced the power-law rule used by NEST's HPC
benchmark and obtained a unimodal distribution in a recurrent balanced
network [4].

## Generic NEST STDP rule

NEST's `stdp_synapse` and `stdp_synapse_hom` implement the Gutig et al.
continuum [5]. Let `x = w / Wmax`, let `Kplus` and `Kminus` be the causal and
anti-causal timing traces sampled for an update, and let `lambda` and `alpha`
be the learning-rate and depression-asymmetry parameters. NEST computes

```text
x <- min(1, x + lambda * (1 - x)^mu_plus * Kplus)
x <- max(0, x - alpha * lambda * x^mu_minus * Kminus)
w <- Wmax * x
```

The exponents select important named cases:

| `mu_plus` | `mu_minus` | NEST documentation name | Boundary behavior |
|---:|---:|---|---|
| 0 | 0 | Song additive STDP | Both update magnitudes remain finite at the bounds; hard clipping promotes a bimodal distribution |
| 1 | 1 | Multiplicative STDP | Potentiation vanishes at `Wmax` and depression vanishes at zero; weights have an interior equilibrium |
| 0 | 1 | van Rossum STDP | Additive potentiation and weight-proportional depression; typically unimodal and positively skewed |
| between 0 and 1 | between 0 and 1 | Gutig STDP | Continuous interpolation between strong competition and soft stabilization |

The implementation is in
`3rdparty/nest-simulator/models/stdp_synapse_hom.h`: `facilitate_()` applies
the `(1 - x)^mu_plus` factor and `depress_()` applies the `x^mu_minus` factor.
In the vendored NEST revision, the default generic rule has
`mu_plus = mu_minus = 1`, so an experiment intended to test additive STDP must
set both exponents to zero explicitly.

The two updates are event-paired rather than applied once per integration
step. On a presynaptic spike, NEST first potentiates for postsynaptic spikes
that occurred since the preceding presynaptic event, using the decayed
presynaptic `Kplus` trace. It then depresses for the new presynaptic spike using
the postsynaptic neuron's current `Kminus` value. `tau_plus` is a common
synapse-model parameter; `tau_minus` belongs to the postsynaptic neuron. The
model ignores the sub-timestep offset of precise spike timestamps.

## Morrison power-law HPC rule

The NEST HPC benchmark uses `stdp_pl_synapse_hom_hpc`, the HPC connection
representation of `stdp_pl_synapse_hom`. This is a different rule from the
generic bounded model above. For each timing event it applies

```text
potentiation: w <- w + lambda * w^mu * Kplus
depression:   w <- max(0, w - lambda * alpha * w * Kminus)
```

The benchmark uses `mu = 0.4`, `lambda = 0.1`, and `alpha = 0.0513`.
Potentiation is sublinear in the current weight and depression is
multiplicative. Very weak weights therefore potentiate only weakly, while
depression scales linearly with weight. The resulting drift has an interior
fixed point and yields the intended unimodal distribution.

The rule is implemented in
`3rdparty/nest-simulator/models/stdp_pl_synapse_hom.h`. Its `facilitate_()`
adds `lambda * pow(w, mu) * Kplus`; its `depress_()` subtracts
`lambda * alpha * w * Kminus`.

The name encodes three separate choices: `pl` selects this power-law rule,
`hom` stores `tau_plus`, `lambda`, `alpha`, and `mu` once as common properties
rather than once per connection, and `_hpc` selects NEST's compact connection
representation. `_hpc` changes storage and traversal, not the mathematical
update. Unlike the generic `stdp_synapse_hom`, this rule has no `Wmax` and no
`mu_plus`/`mu_minus`; it has the single potentiation exponent `mu`, while its
depression is always linear in `w`.

## Expected Brunel-network behavior

For the Morrison rule at an adequate network scale, the intended state is
low-rate asynchronous irregular firing in mutual equilibrium with a unimodal
E-to-E weight distribution. The NEST benchmark documentation requires a
reported rate below 10 spikes/s as a basic check. Morrison et al. reported an
approximately Gaussian distribution near `45.65 +/- 3.99 pA`, a firing rate
near `8.8 Hz`, and continuing turnover of which individual synapses are
strong; spontaneous persistent assemblies did not form [4].

For the additive comparison, a successful qualitative result is an initially
interior distribution separating into two modes near zero and `Wmax`. The
distribution classifier must inspect boundary occupancy and histogram modes,
not only its mean and standard deviation.

There is also an important failure mode. If depression undercompensates in the
Morrison network, positive feedback between correlations and weights can make
the firing rate jump, split the distribution into several clusters, and drive
the network into synchronous bursts separated by silence [4]. This is not the
stable additive-STDP bimodal equilibrium. A run is classified as unstable if
weight clustering accompanies a large rate increase, population synchrony,
or severe simulation slowdown.

## Sources

1. Morrison A, Diesmann M, Gerstner W (2008), [Phenomenological models of synaptic plasticity based on spike timing](https://pmc.ncbi.nlm.nih.gov/articles/PMC2799003/).
2. Song S, Miller KD, Abbott LF (2000), [Competitive Hebbian learning through spike-timing-dependent synaptic plasticity](https://doi.org/10.1038/78829).
3. van Rossum MCW, Bi GQ, Turrigiano GG (2000), [Stable Hebbian learning from spike timing-dependent plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC6773092/).
4. Morrison A, Aertsen A, Diesmann M (2007), [Spike-timing-dependent plasticity in balanced random networks](https://doi.org/10.1162/neco.2007.19.6.1437).
5. Gutig R, Aharonov R, Rotter S, Sompolinsky H (2003), [Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003).
6. NEST 3.5, [Random balanced network HPC benchmark](https://nest-simulator.readthedocs.io/en/v3.5/auto_examples/hpc_benchmark.html).
