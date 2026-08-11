# Analysis: Current Rust Implementation vs Reference (Diehl & Cook 2015)

## Summary

The Rust SNN peaked at **67.5% accuracy** then decayed to ~55% over 3 epochs with 400 neurons on MNIST. The reference implementation achieves ~95% on MNIST. The gap stems from 10 fundamental algorithmic differences.

## Critical Differences

### 1. Neuron Model: Current-Based vs Conductance-Based
**Reference**: Conductance-based LIF with reversal potentials:
```
dv/dt = ((v_rest - v) + ge*(0 - v) + gi*(E_inh - v)) / tau
```
Excitatory synaptic current depends on `ge * (E_exc - v)` — voltage-dependent, shunting inhibition.

**Rust**: Simple current-based LIF:
```
v += input; v += (rest - v) / tau
```
Input directly added to voltage. No reversal potentials, no conductance dynamics.

**Impact**: The conductance-based model provides automatic gain control through voltage-dependent currents. As v approaches E_exc=0mV, excitatory drive diminishes. This prevents runaway excitation and enables balanced E/I dynamics.

### 2. ODE Integration at dt=0.5ms
**Reference**: Euler integration at dt=0.5ms, continuous dynamics within each timestep.

**Rust**: Discrete 1-tick events; no sub-step integration.

**Impact**: The reference's fine-grained integration captures conductance decay dynamics within each step.

### 3. Conductance Dynamics (ge, gi)
**Reference**: `dge/dt = -ge/1ms`, `dgi/dt = -gi/2ms` — conductances decay exponentially after each spike. A presynaptic spike adds the weight to ge/gi, then the conductance influences voltage over multiple timesteps.

**Rust**: Weight directly added to voltage on spike tick. No temporal spread.

**Impact**: Conductance dynamics provide temporal integration — a single spike's effect lasts ~2-5 time constants. This gives the network much richer temporal dynamics and better integration of spike timing.

### 4. STDP Rule: Pair-Based vs Trace-Based Triplet
**Reference** (trace-based triplet):
- Pre-spike: `pre_trace = 1; Δw = -nu_pre * post1_trace`
- Post-spike: `post2before = post2_trace; Δw = +nu_post * pre_trace * post2before; post1_trace = 1; post2_trace = 1`
- Traces decay: `d(trace)/dt = -trace / tau_trace`

This is a **triplet rule** where potentiation depends on `pre * post2` (previous postsynaptic activity).

**Rust** (pair-based):
- Pre-spike: `Δw = -lr * exp(-Δt/tau)` based on elapsed time since last post-spike
- Post-spike: `Δw = +lr * exp(-Δt/tau)` based on elapsed time since last pre-spike

**Impact**: The triplet rule better captures rate-dependent plasticity. The `post2` trace means potentiation requires repeated postsynaptic activity, preventing spurious LTP from single coincidences.

### 5. Weight Normalization
**Reference**: Before each training sample, normalize each column of input→exc weights so the column sum equals **78.0**. Simple multiplicative rescaling.

**Rust**: After STDP updates, renormalize touched columns using L1 or L2 norm to a target gain. Also has slow synaptic scaling.

**Impact**: The reference's normalization is simpler and more aggressive — it forces competition between synapses every sample. The constant sum constraint (78.0) ensures stable total drive to each neuron.

### 6. Threshold Adaptation (Theta)
**Reference**:
- `dtheta/dt = -theta / 1e7ms` (extremely slow decay, essentially permanent)
- On spike: `theta += 0.05mV`
- Effective threshold: `v > theta - 20mV + (-52mV)` = `v > theta - 72mV`
- Theta starts at 20mV, so initial effective threshold = -52mV
- Over training, active neurons accumulate theta, raising their threshold

**Rust**:
- Homeostasis: `active_threshold += (base - active) / tau` (decays toward base)
- On spike: `active_threshold += homeo_inc`
- `homeo_inc = 2.0`, `tau = 5.0` — very fast dynamics

**Impact**: The reference's theta creates a nearly permanent record of neuron activity — neurons that fire a lot become harder to activate, forcing diversity. The Rust version's fast decay allows threshold to recover quickly, providing less diversity pressure.

### 7. Input Encoding
**Reference**: `rate = (pixel_value / 8) * intensity` where pixel_value ∈ [0,255], intensity starts at 2. So max rate = 255/8 * 2 = 63.75 Hz. The PoissonGroup generates spikes at this rate with dt=0.5ms.

**Rust**: `rate = (pixel_value / 255) * poisson_rate` where poisson_rate=0.1. This gives max firing probability 0.1 per tick.

**Impact**: Different effective input rates and dynamics.

### 8. Retry Mechanism
**Reference**: If fewer than 5 total spikes across all excitatory neurons during 350ms presentation, increment `intensity += 1` and retry. Neural state is NOT reset between retries (only rest phase runs).

**Rust**: Reset and retry with increased Poisson rate. Different retry behavior.

### 9. Simulation Timing
**Reference**: dt=0.5ms, presentation=350ms (700 steps), rest=150ms (300 steps).

**Rust**: 100 ticks per sample, 40 rest ticks. Much shorter presentation.

### 10. Synaptic Delays
**Reference**: Random delays uniformly in [0, 10ms) on input→exc connections.

**Rust**: No delays.

**Impact**: Delays spread temporal correlations and may help pattern recognition.

## Conclusion

The most impactful differences are:
1. **Conductance-based neuron model** — provides natural gain control
2. **Trace-based triplet STDP** — more selective potentiation
3. **Weight normalization** — simple, aggressive, stable
4. **Threshold adaptation** — nearly permanent, forces diversity

Aligning these should bring accuracy close to the reference.
