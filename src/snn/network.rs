use serde::{Deserialize, Serialize};

use crate::snn::neurons::Lif;
use crate::snn::weight::Weight;

use super::neurons::Neuron;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedforwardNormalization {
    None,
    L1,
    L2,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedforwardHomeostasisConfig {
    pub normalization: FeedforwardNormalization,
    pub post_renorm_gain: f32,
    pub slow_scaling_rate: f32,
    pub slow_scaling_target_rate: f32,
    pub slow_scaling_alpha: f32,
    pub slow_scaling_min_gain: f32,
    pub slow_scaling_max_gain: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MnistNeuronConfig {
    pub rest_potential: f32,
    pub reset_potential: f32,
    pub membrane_tau: f32,
    pub threshold: f32,
    pub refractory_ticks: usize,
    pub e_exc: f32,
    pub e_inh: f32,
    pub ge_decay: f32,
    pub gi_decay: f32,
    pub theta_init: f32,
    pub theta_plus: f32,
    pub theta_tc: f32,
    // Inhibitory neuron overrides (None = use excitatory value)
    pub inhibitory_rest_potential: Option<f32>,
    pub inhibitory_reset_potential: Option<f32>,
    pub inhibitory_membrane_tau: Option<f32>,
    pub inhibitory_threshold: Option<f32>,
    pub inhibitory_refractory_ticks: Option<usize>,
    pub inhibitory_e_inh: Option<f32>,
    pub inhibitory_theta_init: Option<f32>,
    pub inhibitory_theta_plus: Option<f32>,
    pub inhibitory_theta_tc: Option<f32>,
}

impl Default for MnistNeuronConfig {
    fn default() -> Self {
        Self {
            rest_potential: MNIST_REST_POTENTIAL,
            reset_potential: MNIST_RESET_POTENTIAL,
            membrane_tau: MNIST_TAU,
            threshold: MNIST_THRESHOLD,
            refractory_ticks: MNIST_REFRACTORY_TICKS,
            e_exc: MNIST_E_EXC,
            e_inh: MNIST_E_INH_EXC,
            ge_decay: MNIST_GE_DECAY,
            gi_decay: MNIST_GI_DECAY,
            theta_init: MNIST_THETA_INIT,
            theta_plus: MNIST_THETA_PLUS,
            theta_tc: MNIST_THETA_TC,
            inhibitory_rest_potential: Some(MNIST_INH_REST_POTENTIAL),
            inhibitory_reset_potential: Some(MNIST_INH_RESET_POTENTIAL),
            inhibitory_membrane_tau: Some(MNIST_INH_TAU),
            inhibitory_threshold: Some(MNIST_INH_THRESHOLD),
            inhibitory_refractory_ticks: Some(MNIST_INH_REFRACTORY_TICKS),
            inhibitory_e_inh: Some(MNIST_E_INH_INH),
            inhibitory_theta_init: Some(0.0),
            inhibitory_theta_plus: Some(0.0),
            inhibitory_theta_tc: Some(1e10), // effectively no adaptation
        }
    }
}

impl MnistNeuronConfig {
    fn inhibitory_rest_potential(self) -> f32 {
        self.inhibitory_rest_potential.unwrap_or(self.rest_potential)
    }
    fn inhibitory_reset_potential(self) -> f32 {
        self.inhibitory_reset_potential.unwrap_or(self.reset_potential)
    }
    fn inhibitory_membrane_tau(self) -> f32 {
        self.inhibitory_membrane_tau.unwrap_or(self.membrane_tau)
    }
    fn inhibitory_threshold(self) -> f32 {
        self.inhibitory_threshold.unwrap_or(self.threshold)
    }
    fn inhibitory_refractory_ticks(self) -> usize {
        self.inhibitory_refractory_ticks.unwrap_or(self.refractory_ticks)
    }
    fn inhibitory_e_inh(self) -> f32 {
        self.inhibitory_e_inh.unwrap_or(self.e_inh)
    }
    fn inhibitory_theta_init(self) -> f32 {
        self.inhibitory_theta_init.unwrap_or(self.theta_init)
    }
    fn inhibitory_theta_plus(self) -> f32 {
        self.inhibitory_theta_plus.unwrap_or(self.theta_plus)
    }
    fn inhibitory_theta_tc(self) -> f32 {
        self.inhibitory_theta_tc.unwrap_or(self.theta_tc)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MnistStdpConfig {
    pub weight_max: Option<f32>,
    pub lr_plus: Option<f32>,
    pub lr_minus: Option<f32>,
    pub tau_plus: Option<f32>,
    pub tau_minus: Option<f32>,
}

impl MnistStdpConfig {
    fn resolved_weight_max(self, feedforward_homeostasis: FeedforwardHomeostasisConfig) -> f32 {
        self.weight_max.unwrap_or_else(|| match feedforward_homeostasis.normalization {
            FeedforwardNormalization::None => MNIST_SYNAPSE_WEIGHT,
            FeedforwardNormalization::L1 | FeedforwardNormalization::L2 => feedforward_homeostasis
                .slow_scaling_max_gain
                .max(feedforward_homeostasis.post_renorm_gain)
                .max(MNIST_SYNAPSE_WEIGHT),
        })
    }

    fn resolved_lr_plus(self) -> f32 {
        self.lr_plus.unwrap_or(MNIST_STDP_LR_PLUS)
    }

    fn resolved_lr_minus(self) -> f32 {
        self.lr_minus.unwrap_or(MNIST_STDP_LR_MINUS)
    }

    fn resolved_tau_plus(self) -> f32 {
        self.tau_plus.unwrap_or(MNIST_STDP_TAU_PLUS)
    }

    fn resolved_tau_minus(self) -> f32 {
        self.tau_minus.unwrap_or(MNIST_STDP_TAU_MINUS)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedforwardSparsityStats {
    pub total_slots: usize,
    pub present_synapses: usize,
    pub near_zero_synapses: usize,
}

impl FeedforwardSparsityStats {
    pub fn nonzero_synapses(self) -> usize {
        self.present_synapses.saturating_sub(self.near_zero_synapses)
    }

    pub fn present_ratio(self) -> f32 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.present_synapses as f32 / self.total_slots as f32
        }
    }

    pub fn nonzero_ratio(self) -> f32 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.nonzero_synapses() as f32 / self.total_slots as f32
        }
    }

    pub fn near_zero_present_ratio(self) -> f32 {
        if self.present_synapses == 0 {
            0.0
        } else {
            self.near_zero_synapses as f32 / self.present_synapses as f32
        }
    }
}

impl FeedforwardHomeostasisConfig {
    fn scaling_enabled(self) -> bool {
        self.normalization != FeedforwardNormalization::None
            && self.slow_scaling_rate > 0.0
            && self.slow_scaling_alpha > 0.0
    }
}


// --- SynapseGroup: Weight matrix + shared STDP parameters ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynapseGroup {
    pub weights: Weight,
    pub max_weight: f32,
    pub min_weight: f32,
    pub lr_plus: f32,
    pub lr_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    /// Second post-synaptic time constant for triplet STDP (tc_post_2_ee).
    /// Potentiation is gated by `exp(-post_elapsed / tau_post2)`, so a neuron
    /// only strongly potentiates when it fires in bursts.  Set to 0 to disable
    /// (falls back to standard pair-based STDP).
    pub tau_post2: f32,
    pub plastic: bool,
}

// --- Tracker ---

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tracker {
    pub last_fire: usize,
}

impl Tracker {
    fn reset(&mut self) {
        self.last_fire = usize::MAX >> 1;
    }

    fn tick(&mut self, fired: bool) {
        if fired {
            self.last_fire = 0;
        } else {
            self.last_fire += 1;
        }
    }
}

// --- Network ---

/// Spiking neural network with three synapse groups:
/// - feedforward: Input → Excitatory (plastic STDP)
/// - e_to_i: Excitatory → Inhibitory
/// - i_to_e: Inhibitory → Excitatory
///
/// Neuron layout: `[0..num_output_neurons)` = excitatory,
/// `[num_output_neurons..neurons.len())` = inhibitory.
pub struct Network<N: Neuron> {
    num_inputs: usize,
    num_output_neurons: usize,

    neurons: Vec<N>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,

    feedforward: SynapseGroup,
    e_to_i: SynapseGroup,
    i_to_e: SynapseGroup,

    feedforward_homeostasis: FeedforwardHomeostasisConfig,
    feedforward_post_gains: Vec<f32>,
    feedforward_rate_ema: Vec<f32>,

    // Feedforward delay support: per-synapse random delays (0..max_delay ticks).
    // Indexed same as feedforward.weights data: [pre * num_post + post].
    feedforward_delays: Vec<u8>,
    // Circular buffer of pending ge deliveries.  delay_buf[slot][post] accumulates
    // weights from spikes scheduled to arrive when the cursor reaches that slot.
    delay_buf: Vec<Vec<f32>>,
    delay_buf_cursor: usize,

    // Reusable per-tick buffers (avoid allocation in hot loop)
    buf_exc_ge: Vec<f32>,       // feedforward excitatory conductance for exc neurons
    buf_exc_gi: Vec<f32>,       // inhibitory conductance for exc neurons (from i_to_e)
    buf_inh_ge: Vec<f32>,       // excitatory conductance for inh neurons (from e_to_i)
    buf_fired_neurons: Vec<bool>,
    buf_touched_posts: Vec<bool>,
    buf_fired_inputs: Vec<usize>,
}

#[derive(Clone)]
pub struct NetworkRuntimeState<N: Clone> {
    neurons: Vec<N>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,
    delay_buf: Vec<Vec<f32>>,
    delay_buf_cursor: usize,
}

// --- Checkpoint ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MnistNetworkCheckpoint {
    num_inputs: usize,
    num_output_neurons: usize,
    neurons: Vec<Lif>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,
    feedforward: SynapseGroup,
    e_to_i: SynapseGroup,
    i_to_e: SynapseGroup,
    feedforward_homeostasis: FeedforwardHomeostasisConfig,
    feedforward_post_gains: Vec<f32>,
    feedforward_rate_ema: Vec<f32>,
    feedforward_delays: Vec<u8>,
    delay_buf: Vec<Vec<f32>>,
    delay_buf_cursor: usize,
}

// --- Network implementation ---

impl<N: Neuron> Network<N> {
    fn summarize(values: &[f32]) -> Option<(f32, f32, f32)> {
        if values.is_empty() {
            return None;
        }

        let min_value = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let avg_value = values.iter().sum::<f32>() / values.len() as f32;
        Some((min_value, avg_value, max_value))
    }

    fn renormalize_postsynaptic_feedforward_weights(&mut self, post: usize) {
        if self.feedforward_homeostasis.normalization == FeedforwardNormalization::None {
            return;
        }

        let target_gain = self.feedforward_post_gains[post];
        let min_w = self.feedforward.min_weight;
        let max_w = self.feedforward.max_weight;

        // Column-sum normalization: scale all weights so their sum equals target_gain.
        let col_sum: f32 = self.feedforward.weights.iter_post(post)
            .map(|(_pre, w)| w)
            .sum();

        if col_sum.abs() < 1e-8 {
            return;
        }

        let scale = target_gain / col_sum;
        for (_pre, w) in self.feedforward.weights.iter_post_mut(post) {
            *w = (*w * scale).clamp(min_w, max_w);
        }
    }

    pub fn update_slow_synaptic_scaling(&mut self, sample_spike_counts: &[usize], per_sample_ticks: usize) {
        assert!(sample_spike_counts.len() == self.num_output_neurons);

        if !self.feedforward_homeostasis.scaling_enabled() {
            return;
        }

        let alpha = self.feedforward_homeostasis.slow_scaling_alpha;
        let scaling_rate = self.feedforward_homeostasis.slow_scaling_rate;
        let target_rate = self.feedforward_homeostasis.slow_scaling_target_rate;
        let min_gain = self.feedforward_homeostasis.slow_scaling_min_gain;
        let max_gain = self.feedforward_homeostasis.slow_scaling_max_gain;

        for post in 0..self.num_output_neurons {
            let observed_rate = sample_spike_counts[post] as f32 / per_sample_ticks as f32;
            self.feedforward_rate_ema[post] += alpha * (observed_rate - self.feedforward_rate_ema[post]);

            let old_gain = self.feedforward_post_gains[post];
            let new_gain = (old_gain * (scaling_rate * (target_rate - self.feedforward_rate_ema[post])).exp())
                .clamp(min_gain, max_gain);

            if (new_gain - old_gain).abs() > 1e-6 {
                self.feedforward_post_gains[post] = new_gain;
                self.renormalize_postsynaptic_feedforward_weights(post);
            }
        }
    }

    pub fn feedforward_gain_stats(&self) -> Option<(f32, f32, f32)> {
        if self.feedforward_homeostasis.normalization == FeedforwardNormalization::None {
            None
        } else {
            Self::summarize(&self.feedforward_post_gains)
        }
    }

    pub fn slow_scaling_rate_ema_stats(&self) -> Option<(f32, f32, f32)> {
        if !self.feedforward_homeostasis.scaling_enabled() {
            None
        } else {
            Self::summarize(&self.feedforward_rate_ema)
        }
    }

    pub fn feedforward_sparsity_stats(&self, near_zero_threshold: f32) -> FeedforwardSparsityStats {
        assert!(near_zero_threshold >= 0.0);

        let mut present_synapses = 0usize;
        let mut near_zero_synapses = 0usize;
        self.feedforward.weights.for_each_weight(|_pre, _post, w| {
            present_synapses += 1;
            if w.abs() <= near_zero_threshold {
                near_zero_synapses += 1;
            }
        });

        FeedforwardSparsityStats {
            total_slots: self.num_inputs * self.num_output_neurons,
            present_synapses,
            near_zero_synapses,
        }
    }

    /// Normalize all feedforward weight columns to their target gain.
    /// Call this before each training sample (reference: normalize_weights).
    pub fn normalize_all_feedforward_weights(&mut self) {
        if self.feedforward_homeostasis.normalization == FeedforwardNormalization::None {
            return;
        }
        for post in 0..self.num_output_neurons {
            self.renormalize_postsynaptic_feedforward_weights(post);
        }
    }

    pub fn tick(&mut self, inputs: &[bool], biases: &[f32], plasticity_enabled: bool, mut fire_tracker: Option<&mut [usize]>) {
        assert!(inputs.len() == self.num_inputs);
        assert!(biases.len() == self.num_output_neurons);
        if let Some(ref ft) = fire_tracker {
            assert!(ft.len() == self.num_output_neurons);
        }

        let n = self.num_output_neurons;
        let num_neurons = self.neurons.len();
        let num_inh = num_neurons - n;

        // 1. Collect fired inputs
        self.buf_fired_inputs.clear();
        for (i, &f) in inputs.iter().enumerate() {
            if f { self.buf_fired_inputs.push(i); }
        }

        // 2. Schedule feedforward spikes into delay buffer
        let delay_buf_len = self.delay_buf.len();
        for &src in &self.buf_fired_inputs {
            let row_offset = src * n;
            for post in 0..n {
                let w = self.feedforward.weights.get(src, post);
                if w != 0.0 {
                    let d = self.feedforward_delays[row_offset + post] as usize;
                    let slot = (self.delay_buf_cursor + d) % delay_buf_len;
                    self.delay_buf[slot][post] += w;
                }
            }
        }

        // Pop current delay slot → buf_exc_ge (spikes arriving NOW)
        self.buf_exc_ge[..n].copy_from_slice(&self.delay_buf[self.delay_buf_cursor][..n]);
        self.delay_buf[self.delay_buf_cursor].fill(0.0);
        self.delay_buf_cursor = (self.delay_buf_cursor + 1) % delay_buf_len;

        // 3. Accumulate e_to_i: Exc -> Inh (excitatory conductance on inh neurons)
        self.buf_inh_ge[..num_inh].fill(0.0);
        for i in 0..n {
            if self.neuron_trackers[i].last_fire == 0 {
                self.e_to_i.weights.accumulate_pre(i, &mut self.buf_inh_ge[..num_inh]);
            }
        }

        // 4. Accumulate i_to_e: Inh -> Exc (inhibitory conductance on exc neurons)
        self.buf_exc_gi[..n].fill(0.0);
        for i in 0..num_inh {
            if self.neuron_trackers[n + i].last_fire == 0 {
                self.i_to_e.weights.accumulate_pre(i, &mut self.buf_exc_gi[..n]);
            }
        }

        // 5. Add conductances and tick neurons
        self.buf_fired_neurons[..num_neurons].fill(false);
        for i in 0..n {
            self.neurons[i].add_ge(self.buf_exc_ge[i]);
            self.neurons[i].add_gi(self.buf_exc_gi[i]);
            let fired = self.neurons[i].tick(biases[i], plasticity_enabled);
            self.buf_fired_neurons[i] = fired;
            if let Some(ref mut ft) = fire_tracker {
                if fired {
                    ft[i] += 1;
                }
            }
        }
        for i in 0..num_inh {
            self.neurons[n + i].add_ge(self.buf_inh_ge[i]);
            let fired = self.neurons[n + i].tick(0.0, plasticity_enabled);
            self.buf_fired_neurons[n + i] = fired;
        }

        // 6. STDP on feedforward synapses only
        if plasticity_enabled && self.feedforward.plastic {
            let lr_minus = self.feedforward.lr_minus;
            let tau_minus = self.feedforward.tau_minus;
            let lr_plus = self.feedforward.lr_plus;
            let tau_plus = self.feedforward.tau_plus;
            let min_weight = self.feedforward.min_weight;
            let max_weight = self.feedforward.max_weight;

            self.buf_touched_posts[..n].fill(false);

            // Depression: pre-spike rule (presynaptic spikes only see previous postsynaptic spikes)
            for idx in 0..self.buf_fired_inputs.len() {
                let i = self.buf_fired_inputs[idx];
                for (post, w) in self.feedforward.weights.iter_pre_mut(i) {
                    if *w == 0.0 && min_weight == 0.0 {
                        continue; // skip absent synapses (sparse connectivity)
                    }
                    let post_elapsed = self.neuron_trackers[post].last_fire.saturating_add(1);
                    let delta = -lr_minus * (-(post_elapsed as f32) / tau_minus).exp();
                    *w = (*w + delta).clamp(min_weight, max_weight);
                    self.buf_touched_posts[post] = true;
                }
            }

            // Potentiation: post-spike rule (postsynaptic spikes can see current input spikes)
            // Triplet STDP: potentiation is gated by post2_trace_before = exp(-post_elapsed / tau_post2),
            // which measures how recently the post neuron fired *before* this spike.
            // This means a neuron only strongly potentiates when it fires in bursts.
            let tau_post2 = self.feedforward.tau_post2;
            for j in 0..n {
                if self.buf_fired_neurons[j] {
                    // post2_trace_before: value of the slow post trace BEFORE this spike sets it to 1.
                    // The tracker hasn't been updated yet (step 7), so last_fire gives ticks
                    // since previous fire. Add 1 because tracker was last incremented at end of
                    // previous tick.
                    let post2_gate = if tau_post2 > 0.0 {
                        let post_elapsed_before = self.neuron_trackers[j].last_fire.saturating_add(1);
                        (-(post_elapsed_before as f32) / tau_post2).exp()
                    } else {
                        1.0 // tau_post2 == 0 disables triplet gating (standard pair STDP)
                    };
                    for (pre, w) in self.feedforward.weights.iter_post_mut(j) {
                        if *w == 0.0 && min_weight == 0.0 {
                            continue; // skip absent synapses (sparse connectivity)
                        }
                        let pre_elapsed = if inputs[pre] {
                            0
                        } else {
                            self.input_trackers[pre].last_fire.saturating_add(1)
                        };
                        let delta = lr_plus * (-(pre_elapsed as f32) / tau_plus).exp() * post2_gate;
                        *w = (*w + delta).clamp(min_weight, max_weight);
                        self.buf_touched_posts[j] = true;
                    }
                }
            }

            // Weight normalization is applied once per sample (before each
            // presentation) via normalize_all_feedforward_weights(), matching
            // the reference.  During a sample, STDP can temporarily push
            // column sums above/below the target, amplifying WTA dynamics.
        }

        // 7. Update trackers
        for (t, &input) in self.input_trackers.iter_mut().zip(inputs.iter()) {
            t.tick(input);
        }
        for i in 0..num_neurons {
            self.neuron_trackers[i].tick(self.buf_fired_neurons[i]);
        }
    }

    pub fn reset_neurons(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset();
        }
        for tracker in self.input_trackers.iter_mut() {
            tracker.reset();
        }
        for tracker in self.neuron_trackers.iter_mut() {
            tracker.reset();
        }
        for slot in self.delay_buf.iter_mut() {
            slot.fill(0.0);
        }
        self.delay_buf_cursor = 0;
    }
}

impl<N: Neuron + Clone> Network<N> {
    fn snapshot_runtime_state(&self) -> NetworkRuntimeState<N> {
        NetworkRuntimeState {
            neurons: self.neurons.clone(),
            input_trackers: self.input_trackers.clone(),
            neuron_trackers: self.neuron_trackers.clone(),
            delay_buf: self.delay_buf.clone(),
            delay_buf_cursor: self.delay_buf_cursor,
        }
    }

    fn restore_runtime_state(&mut self, state: NetworkRuntimeState<N>) {
        self.neurons = state.neurons;
        self.input_trackers = state.input_trackers;
        self.neuron_trackers = state.neuron_trackers;
        self.delay_buf = state.delay_buf;
        self.delay_buf_cursor = state.delay_buf_cursor;
    }
}

impl<N: Neuron> Network<N> {
    fn new_from(
        num_inputs: usize,
        num_output_neurons: usize,
        neurons: Vec<N>,
        feedforward: SynapseGroup,
        e_to_i: SynapseGroup,
        i_to_e: SynapseGroup,
        feedforward_homeostasis: FeedforwardHomeostasisConfig,
        feedforward_delays: Vec<u8>,
        max_delay: usize,
    ) -> Self {
        let num_neurons = neurons.len();
        let num_inh = num_neurons - num_output_neurons;

        assert!(num_output_neurons <= num_neurons);
        assert_eq!(feedforward.weights.num_pre(), num_inputs);
        assert_eq!(feedforward.weights.num_post(), num_output_neurons);
        assert_eq!(e_to_i.weights.num_pre(), num_output_neurons);
        assert_eq!(e_to_i.weights.num_post(), num_inh);
        assert_eq!(i_to_e.weights.num_pre(), num_inh);
        assert_eq!(i_to_e.weights.num_post(), num_output_neurons);
        assert!(feedforward_homeostasis.post_renorm_gain > 0.0);
        assert!(feedforward_homeostasis.slow_scaling_alpha >= 0.0 && feedforward_homeostasis.slow_scaling_alpha <= 1.0);
        assert!(feedforward_homeostasis.slow_scaling_min_gain > 0.0);
        assert!(feedforward_homeostasis.slow_scaling_max_gain >= feedforward_homeostasis.slow_scaling_min_gain);
        assert_eq!(feedforward_delays.len(), feedforward.weights.num_synapses());

        let input_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; num_inputs];
        let neuron_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; num_neurons];
        let feedforward_post_gains = vec![feedforward_homeostasis.post_renorm_gain; num_output_neurons];
        let feedforward_rate_ema = vec![feedforward_homeostasis.slow_scaling_target_rate; num_output_neurons];

        let delay_buf_len = max_delay + 1;
        let delay_buf = vec![vec![0.0; num_output_neurons]; delay_buf_len];

        let mut network = Network {
            num_inputs,
            num_output_neurons,
            neurons,
            input_trackers,
            neuron_trackers,
            feedforward,
            e_to_i,
            i_to_e,
            feedforward_homeostasis,
            feedforward_post_gains,
            feedforward_rate_ema,
            feedforward_delays,
            delay_buf,
            delay_buf_cursor: 0,
            buf_exc_ge: vec![0.0; num_output_neurons],
            buf_exc_gi: vec![0.0; num_output_neurons],
            buf_inh_ge: vec![0.0; num_inh],
            buf_fired_neurons: vec![false; num_neurons],
            buf_touched_posts: vec![false; num_output_neurons],
            buf_fired_inputs: Vec::with_capacity(num_inputs),
        };

        if feedforward_homeostasis.normalization != FeedforwardNormalization::None {
            for post in 0..num_output_neurons {
                network.renormalize_postsynaptic_feedforward_weights(post);
            }
        }

        network
    }
}

// --- PoissonInputNetwork ---

pub struct PoissonInputNetwork<N: Neuron> {
    network: Network<N>,
    buf_poisson_inputs: Vec<bool>,
}

impl<N: Neuron> PoissonInputNetwork<N> {
    pub fn tick(&mut self, rates: &[f32], biases: &[f32], plasticity_enabled: bool, fire_tracker: Option<&mut [usize]>) {
        self.buf_poisson_inputs.clear();
        self.buf_poisson_inputs.extend(rates.iter().map(|&r| rand::random::<f32>() < r));
        self.network.tick(&self.buf_poisson_inputs, biases, plasticity_enabled, fire_tracker);
    }

    pub fn reset_neurons(&mut self) {
        self.network.reset_neurons();
    }

    pub fn update_slow_synaptic_scaling(&mut self, sample_spike_counts: &[usize], per_sample_ticks: usize) {
        self.network.update_slow_synaptic_scaling(sample_spike_counts, per_sample_ticks);
    }

    pub fn feedforward_gain_stats(&self) -> Option<(f32, f32, f32)> {
        self.network.feedforward_gain_stats()
    }

    pub fn slow_scaling_rate_ema_stats(&self) -> Option<(f32, f32, f32)> {
        self.network.slow_scaling_rate_ema_stats()
    }

    pub fn feedforward_sparsity_stats(&self, near_zero_threshold: f32) -> FeedforwardSparsityStats {
        self.network.feedforward_sparsity_stats(near_zero_threshold)
    }

    pub fn normalize_all_feedforward_weights(&mut self) {
        self.network.normalize_all_feedforward_weights();
    }
}

impl<N: Neuron + Clone> PoissonInputNetwork<N> {
    pub fn snapshot_runtime_state(&self) -> NetworkRuntimeState<N> {
        self.network.snapshot_runtime_state()
    }

    pub fn restore_runtime_state(&mut self, state: NetworkRuntimeState<N>) {
        self.network.restore_runtime_state(state);
    }
}

impl<N: Neuron> From<Network<N>> for PoissonInputNetwork<N> {
    fn from(network: Network<N>) -> Self {
        let num_inputs = network.num_inputs;
        PoissonInputNetwork { network, buf_poisson_inputs: Vec::with_capacity(num_inputs) }
    }
}

// --- Checkpoint serialization (Lif-specific) ---

impl PoissonInputNetwork<Lif> {
    pub fn to_checkpoint(&self) -> MnistNetworkCheckpoint {
        MnistNetworkCheckpoint {
            num_inputs: self.network.num_inputs,
            num_output_neurons: self.network.num_output_neurons,
            neurons: self.network.neurons.clone(),
            input_trackers: self.network.input_trackers.clone(),
            neuron_trackers: self.network.neuron_trackers.clone(),
            feedforward: self.network.feedforward.clone(),
            e_to_i: self.network.e_to_i.clone(),
            i_to_e: self.network.i_to_e.clone(),
            feedforward_homeostasis: self.network.feedforward_homeostasis,
            feedforward_post_gains: self.network.feedforward_post_gains.clone(),
            feedforward_rate_ema: self.network.feedforward_rate_ema.clone(),
            feedforward_delays: self.network.feedforward_delays.clone(),
            delay_buf: self.network.delay_buf.clone(),
            delay_buf_cursor: self.network.delay_buf_cursor,
        }
    }

    pub fn from_checkpoint(checkpoint: MnistNetworkCheckpoint) -> Self {
        let num_neurons = checkpoint.neurons.len();
        let num_inh = num_neurons - checkpoint.num_output_neurons;
        let num_inputs = checkpoint.num_inputs;
        let num_output = checkpoint.num_output_neurons;

        assert!(checkpoint.num_output_neurons <= num_neurons);
        assert_eq!(checkpoint.input_trackers.len(), num_inputs);
        assert_eq!(checkpoint.neuron_trackers.len(), num_neurons);
        assert_eq!(checkpoint.feedforward_post_gains.len(), num_output);
        assert_eq!(checkpoint.feedforward_rate_ema.len(), num_output);
        assert_eq!(checkpoint.feedforward.weights.num_pre(), num_inputs);
        assert_eq!(checkpoint.feedforward.weights.num_post(), num_output);
        assert_eq!(checkpoint.e_to_i.weights.num_pre(), num_output);
        assert_eq!(checkpoint.e_to_i.weights.num_post(), num_inh);
        assert_eq!(checkpoint.i_to_e.weights.num_pre(), num_inh);
        assert_eq!(checkpoint.i_to_e.weights.num_post(), num_output);
        assert!(checkpoint.feedforward_homeostasis.post_renorm_gain > 0.0);
        assert!(
            checkpoint.feedforward_homeostasis.slow_scaling_alpha >= 0.0
                && checkpoint.feedforward_homeostasis.slow_scaling_alpha <= 1.0
        );
        assert!(checkpoint.feedforward_homeostasis.slow_scaling_min_gain > 0.0);
        assert!(
            checkpoint.feedforward_homeostasis.slow_scaling_max_gain
                >= checkpoint.feedforward_homeostasis.slow_scaling_min_gain
        );
        assert_eq!(checkpoint.feedforward_delays.len(), checkpoint.feedforward.weights.num_synapses());

        PoissonInputNetwork {
            network: Network {
                num_inputs: checkpoint.num_inputs,
                num_output_neurons: checkpoint.num_output_neurons,
                neurons: checkpoint.neurons,
                input_trackers: checkpoint.input_trackers,
                neuron_trackers: checkpoint.neuron_trackers,
                feedforward: checkpoint.feedforward,
                e_to_i: checkpoint.e_to_i,
                i_to_e: checkpoint.i_to_e,
                feedforward_homeostasis: checkpoint.feedforward_homeostasis,
                feedforward_post_gains: checkpoint.feedforward_post_gains,
                feedforward_rate_ema: checkpoint.feedforward_rate_ema,
                feedforward_delays: checkpoint.feedforward_delays,
                delay_buf: checkpoint.delay_buf,
                delay_buf_cursor: checkpoint.delay_buf_cursor,
                buf_exc_ge: vec![0.0; num_output],
                buf_exc_gi: vec![0.0; num_output],
                buf_inh_ge: vec![0.0; num_inh],
                buf_fired_neurons: vec![false; num_neurons],
                buf_touched_posts: vec![false; num_output],
                buf_fired_inputs: Vec::with_capacity(num_inputs),
            },
            buf_poisson_inputs: Vec::with_capacity(num_inputs),
        }
    }

    /// Diagnostic: dump key network statistics for debugging training dynamics.
    pub fn dump_diagnostics(&self, n_exc: usize) {
        let net = &self.network;
        let thetas: Vec<f64> = net.neurons[..n_exc].iter().map(|n| n.theta).collect();
        let theta_min = thetas.iter().cloned().fold(f64::INFINITY, f64::min);
        let theta_max = thetas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let theta_avg = thetas.iter().sum::<f64>() / thetas.len() as f64;
        let theta_above_72 = thetas.iter().filter(|&&t| t as f32 + net.neurons[0].threshold > net.neurons[0].e_exc).count();

        eprintln!("  THETA: min={:.2} avg={:.2} max={:.2} dead(eff_thresh>E_exc)={} theta_decay={:.15} theta_plus={}",
            theta_min, theta_avg, theta_max, theta_above_72,
            net.neurons[0].theta_decay, net.neurons[0].theta_plus);

        let ge_vals: Vec<f32> = net.neurons[..n_exc].iter().map(|n| n.ge).collect();
        let gi_vals: Vec<f32> = net.neurons[..n_exc].iter().map(|n| n.gi).collect();
        let ge_max = ge_vals.iter().cloned().fold(0.0f32, f32::max);
        let gi_max = gi_vals.iter().cloned().fold(0.0f32, f32::max);
        let ge_nz = ge_vals.iter().filter(|&&g| g > 1e-6).count();
        let gi_nz = gi_vals.iter().filter(|&&g| g > 1e-6).count();
        eprintln!("  GE: max={:.4} nonzero={} | GI: max={:.4} nonzero={}", ge_max, ge_nz, gi_max, gi_nz);

        let mut col_sums = Vec::with_capacity(n_exc);
        let mut col_maxes = Vec::with_capacity(n_exc);
        let mut col_nz = Vec::with_capacity(n_exc);
        for post in 0..n_exc {
            let mut sum = 0.0f32;
            let mut max_w = 0.0f32;
            let mut nz = 0usize;
            for (_pre, w) in net.feedforward.weights.iter_post(post) {
                sum += w;
                if w > max_w { max_w = w; }
                if w > 1e-6 { nz += 1; }
            }
            col_sums.push(sum);
            col_maxes.push(max_w);
            col_nz.push(nz);
        }
        let cs_min = col_sums.iter().cloned().fold(f32::INFINITY, f32::min);
        let cs_max = col_sums.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let cs_avg = col_sums.iter().sum::<f32>() / col_sums.len() as f32;
        let wmax_max = col_maxes.iter().cloned().fold(0.0f32, f32::max);
        let nz_min = *col_nz.iter().min().unwrap_or(&0);
        let nz_max = *col_nz.iter().max().unwrap_or(&0);
        eprintln!("  FF_COL_SUM: min={:.2} avg={:.2} max={:.2} | wmax={:.4} | nz_per_col: {}..{}",
            cs_min, cs_avg, cs_max, wmax_max, nz_min, nz_max);

        let v_vals: Vec<f32> = net.neurons[..n_exc].iter().map(|n| n.v).collect();
        let v_min = v_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = v_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let refrac_count = net.neurons[..n_exc].iter().filter(|n| n.refractory_remaining > 0).count();
        eprintln!("  V: min={:.2} max={:.2} refractory={}", v_min, v_max, refrac_count);
    }
}
// Reference parameters (Diehl & Cook 2015), FTDT discretized.
// 1 tick = 0.5ms (matching reference dt).
// Excitatory neurons
const MNIST_REST_POTENTIAL: f32 = -65.0;  // mV
const MNIST_RESET_POTENTIAL: f32 = -65.0; // mV (same as rest in reference)
const MNIST_TAU: f32 = 200.0;            // membrane time constant: 100ms / 0.5ms = 200 ticks
const MNIST_THRESHOLD: f32 = -72.0;       // mV: v_thresh_e(-52) - offset(20); ref threshold = theta - offset + v_thresh_e
const MNIST_REFRACTORY_TICKS: usize = 10; // 5ms / 0.5ms = 10 ticks
const MNIST_E_EXC: f32 = 0.0;            // excitatory reversal (mV)
const MNIST_E_INH_EXC: f32 = -100.0;     // inhibitory reversal for exc neurons (mV)
const MNIST_GE_DECAY: f32 = 0.6065;      // exp(-0.5/1) = exp(-1/2): tau_ge=1ms, dt=0.5ms → 2 ticks
const MNIST_GI_DECAY: f32 = 0.7788;      // exp(-0.5/2) = exp(-1/4): tau_gi=2ms, dt=0.5ms → 4 ticks
const MNIST_THETA_INIT: f32 = 20.0;      // mV (initial theta)
const MNIST_THETA_PLUS: f32 = 0.05;      // mV (increment on spike)
const MNIST_THETA_TC: f32 = 1e5;         // tc_theta: 5e4ms / 0.5ms = 1e5 ticks (faster decay)
// Inhibitory neurons
const MNIST_INH_REST_POTENTIAL: f32 = -60.0;
const MNIST_INH_RESET_POTENTIAL: f32 = -45.0;
const MNIST_INH_TAU: f32 = 20.0;         // 10ms / 0.5ms = 20 ticks
const MNIST_INH_THRESHOLD: f32 = -40.0;
const MNIST_INH_REFRACTORY_TICKS: usize = 4; // 2ms / 0.5ms = 4 ticks
const MNIST_E_INH_INH: f32 = -85.0;      // inhibitory reversal for inh neurons
// STDP (time constants in ticks: ms / 0.5ms)
const MNIST_STDP_LR_PLUS: f32 = 0.01;
const MNIST_STDP_LR_MINUS: f32 = 0.0001; // nu_ee_pre in reference
const MNIST_STDP_TAU_PLUS: f32 = 40.0;   // tc_pre_ee: 20ms / 0.5ms = 40 ticks
const MNIST_STDP_TAU_MINUS: f32 = 40.0;  // tc_post_1_ee: 20ms / 0.5ms = 40 ticks
const MNIST_STDP_TAU_POST2: f32 = 80.0;  // tc_post_2_ee: 40ms / 0.5ms = 80 ticks (triplet STDP)
// Weights
const MNIST_SYNAPSE_WEIGHT: f32 = 1.0;   // wmax
const MNIST_FEEDFORWARD_INIT_SCALE: f32 = 0.3;  // random * 0.3 + 0.01
const MNIST_FEEDFORWARD_INIT_OFFSET: f32 = 0.01;
const MNIST_REF_E_TO_I_WEIGHT: f32 = 10.4;
const MNIST_REF_I_TO_E_WEIGHT: f32 = 17.0;
// Feedforward delay: reference uses uniform random delay (0, 10ms) = (0, 20 ticks)
const MNIST_MAX_FEEDFORWARD_DELAY: usize = 20; // 10ms / 0.5ms = 20 ticks

pub fn new_mnist_network(
    output_num: usize,
    connection_rate: f32,
    inhibitory_weight: f32,
    excitatory_to_inhibitory_weight: Option<f32>,
    neuron_config: MnistNeuronConfig,
    stdp_config: MnistStdpConfig,
    feedforward_homeostasis: FeedforwardHomeostasisConfig,
) -> PoissonInputNetwork<Lif> {
    assert!(neuron_config.membrane_tau > 0.0);
    assert!(neuron_config.inhibitory_membrane_tau() > 0.0);
    assert!(inhibitory_weight >= 0.0, "i_to_e weight should be non-negative (it's a conductance magnitude)");
    if let Some(weight) = excitatory_to_inhibitory_weight {
        assert!(weight >= 0.0);
    }

    // Build neurons: N excitatory + N inhibitory
    let mut neurons = Vec::with_capacity(output_num * 2);
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(
            neuron_config.rest_potential,
            neuron_config.reset_potential,
            neuron_config.threshold,
            neuron_config.membrane_tau,
            neuron_config.e_exc,
            neuron_config.e_inh,
            neuron_config.ge_decay,
            neuron_config.gi_decay,
            neuron_config.theta_init,
            neuron_config.theta_plus,
            neuron_config.theta_tc,
            neuron_config.refractory_ticks,
        ));
    }
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(
            neuron_config.inhibitory_rest_potential(),
            neuron_config.inhibitory_reset_potential(),
            neuron_config.inhibitory_threshold(),
            neuron_config.inhibitory_membrane_tau(),
            neuron_config.e_exc,           // same excitatory reversal
            neuron_config.inhibitory_e_inh(),
            neuron_config.ge_decay,        // same conductance dynamics
            neuron_config.gi_decay,
            neuron_config.inhibitory_theta_init(),
            neuron_config.inhibitory_theta_plus(),
            neuron_config.inhibitory_theta_tc(),
            neuron_config.inhibitory_refractory_ticks(),
        ));
    }

    let excitatory_to_inhibitory_weight = excitatory_to_inhibitory_weight
        .unwrap_or_else(|| inhibitory_weight * (MNIST_REF_E_TO_I_WEIGHT / MNIST_REF_I_TO_E_WEIGHT));
    let feedforward_max_weight = stdp_config.resolved_weight_max(feedforward_homeostasis);
    let stdp_lr_plus = stdp_config.resolved_lr_plus();
    let stdp_lr_minus = stdp_config.resolved_lr_minus();
    let stdp_tau_plus = stdp_config.resolved_tau_plus();
    let stdp_tau_minus = stdp_config.resolved_tau_minus();

    // Feedforward: Input(784) -> Exc(N), Dense
    // Reference: random * 0.3 + 0.01
    let feedforward_data: Vec<f32> = (0..784 * output_num)
        .map(|_| {
            if rand::random::<f32>() < connection_rate {
                MNIST_FEEDFORWARD_INIT_SCALE * rand::random::<f32>() + MNIST_FEEDFORWARD_INIT_OFFSET
            } else {
                0.0
            }
        })
        .collect();
    let feedforward = SynapseGroup {
        weights: Weight::new_dense(784, output_num, feedforward_data),
        max_weight: feedforward_max_weight,
        min_weight: 0.0,
        lr_plus: stdp_lr_plus,
        lr_minus: stdp_lr_minus,
        tau_plus: stdp_tau_plus,
        tau_minus: stdp_tau_minus,
        tau_post2: MNIST_STDP_TAU_POST2,
        plastic: true,
    };

    // E->I: OneToOne, fixed weight, no plasticity
    let e_to_i = SynapseGroup {
        weights: Weight::new_one_to_one(vec![excitatory_to_inhibitory_weight; output_num]),
        max_weight: excitatory_to_inhibitory_weight,
        min_weight: excitatory_to_inhibitory_weight,
        lr_plus: 0.0,
        lr_minus: 0.0,
        tau_plus: 1.0,
        tau_minus: 1.0,
        tau_post2: 0.0,
        plastic: false,
    };

    // I->E: Dense(N x N) with zero diagonal, fixed weight, no plasticity
    let mut i_to_e_data = vec![inhibitory_weight; output_num * output_num];
    for i in 0..output_num {
        i_to_e_data[i * output_num + i] = 0.0; // no self-connection
    }
    let i_to_e = SynapseGroup {
        weights: Weight::new_dense(output_num, output_num, i_to_e_data),
        max_weight: inhibitory_weight,
        min_weight: inhibitory_weight,
        lr_plus: 0.0,
        lr_minus: 0.0,
        tau_plus: 1.0,
        tau_minus: 1.0,
        tau_post2: 0.0,
        plastic: false,
    };

    // Random per-synapse feedforward delays: uniform [0, MNIST_MAX_FEEDFORWARD_DELAY]
    let feedforward_delays: Vec<u8> = (0..784 * output_num)
        .map(|_| (rand::random::<f32>() * (MNIST_MAX_FEEDFORWARD_DELAY as f32 + 1.0)) as u8)
        .map(|d| d.min(MNIST_MAX_FEEDFORWARD_DELAY as u8))
        .collect();

    let network = Network::new_from(
        784, output_num, neurons, feedforward, e_to_i, i_to_e,
        feedforward_homeostasis, feedforward_delays, MNIST_MAX_FEEDFORWARD_DELAY,
    );
    network.into()
}

#[cfg(test)]
mod tests {
    use super::{
        FeedforwardHomeostasisConfig, FeedforwardNormalization, MnistNeuronConfig, MnistStdpConfig,
        Network, PoissonInputNetwork, SynapseGroup, new_mnist_network,
    };
    use crate::snn::neurons::Neuron;
    use crate::snn::weight::Weight;

    #[derive(Clone, Default)]
    struct SilentNeuron;

    impl Neuron for SilentNeuron {
        fn tick(&mut self, _input: f32, _plasticity_enabled: bool) -> bool {
            false
        }

        fn reset(&mut self) {}
        fn add_ge(&mut self, _ge: f32) {}
        fn add_gi(&mut self, _gi: f32) {}
    }

    fn homeostasis_config(normalization: FeedforwardNormalization, post_renorm_gain: f32) -> FeedforwardHomeostasisConfig {
        FeedforwardHomeostasisConfig {
            normalization,
            post_renorm_gain,
            slow_scaling_rate: 0.0,
            slow_scaling_target_rate: 0.2,
            slow_scaling_alpha: 0.01,
            slow_scaling_min_gain: 0.25,
            slow_scaling_max_gain: 4.0,
        }
    }

    fn dummy_synapse_group(weights: Weight) -> SynapseGroup {
        SynapseGroup {
            weights,
            max_weight: f32::MAX,
            min_weight: f32::MIN,
            lr_plus: 0.0,
            lr_minus: 0.0,
            tau_plus: 1.0,
            tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        }
    }

    /// Build a minimal test network: `num_inputs` inputs, `num_exc` excitatory
    /// SilentNeurons, `num_exc` inhibitory SilentNeurons, with the given
    /// feedforward weights and zero lateral weights.  All delays are zero.
    fn test_network(
        num_inputs: usize,
        num_exc: usize,
        feedforward_weights: Vec<f32>,
        homeostasis: FeedforwardHomeostasisConfig,
    ) -> Network<SilentNeuron> {
        let num_synapses = num_inputs * num_exc;
        let feedforward = SynapseGroup {
            weights: Weight::new_dense(num_inputs, num_exc, feedforward_weights),
            max_weight: f32::MAX,
            min_weight: 0.0,
            lr_plus: 0.0,
            lr_minus: 0.0,
            tau_plus: 1.0,
            tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        let e_to_i = dummy_synapse_group(Weight::new_one_to_one(vec![0.0; num_exc]));
        let i_to_e = dummy_synapse_group(Weight::new_dense(num_exc, num_exc, vec![0.0; num_exc * num_exc]));
        let neurons = vec![SilentNeuron; num_exc * 2];
        let delays = vec![0u8; num_synapses];

        Network::new_from(num_inputs, num_exc, neurons, feedforward, e_to_i, i_to_e, homeostasis, delays, 0)
    }

    #[test]
    fn new_from_renormalizes_l1_feedforward_weights_to_target_gain() {
        let network = test_network(2, 1, vec![0.2, 0.3], homeostasis_config(FeedforwardNormalization::L1, 2.0));
        let weights: Vec<f32> = network.feedforward.weights.iter_post(0).map(|(_pre, w)| w).collect();
        let l1_norm: f32 = weights.iter().sum();
        assert!((l1_norm - 2.0).abs() < 1e-6);
    }

    #[test]
    fn new_from_renormalizes_l2_feedforward_weights_to_target_gain() {
        // With column-sum normalization, L2 mode also uses sum-to-target
        let network = test_network(2, 1, vec![0.2, 0.3], homeostasis_config(FeedforwardNormalization::L2, 2.0));
        let weights: Vec<f32> = network.feedforward.weights.iter_post(0).map(|(_pre, w)| w).collect();
        let col_sum: f32 = weights.iter().sum();
        assert!((col_sum - 2.0).abs() < 1e-6);
    }

    #[test]
    fn slow_scaling_updates_target_gain_and_preserves_l1_budget() {
        let feedforward = SynapseGroup {
            weights: Weight::new_dense(2, 1, vec![0.2, 0.3]),
            max_weight: f32::MAX,
            min_weight: 0.0,
            lr_plus: 0.0,
            lr_minus: 0.0,
            tau_plus: 1.0,
            tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        let e_to_i = dummy_synapse_group(Weight::new_one_to_one(vec![0.0]));
        let i_to_e = dummy_synapse_group(Weight::new_dense(1, 1, vec![0.0]));
        let neurons = vec![SilentNeuron; 2];

        let mut network = Network::new_from(
            2,
            1,
            neurons,
            feedforward,
            e_to_i,
            i_to_e,
            FeedforwardHomeostasisConfig {
                normalization: FeedforwardNormalization::L1,
                post_renorm_gain: 1.0,
                slow_scaling_rate: 1.0,
                slow_scaling_target_rate: 0.5,
                slow_scaling_alpha: 1.0,
                slow_scaling_min_gain: 0.25,
                slow_scaling_max_gain: 4.0,
            },
            vec![0u8; 2],
            0,
        );

        network.update_slow_synaptic_scaling(&[0], 1);

        let weights: Vec<f32> = network.feedforward.weights.iter_post(0).map(|(_pre, w)| w).collect();
        let l1_norm: f32 = weights.iter().sum();
        let expected_gain = 0.5f32.exp();

        assert!((l1_norm - expected_gain).abs() < 1e-6);
    }

    #[test]
    fn feedforward_sparsity_stats_counts_all_feedforward_weights() {
        let network = test_network(2, 1, vec![1e-5, 0.25], homeostasis_config(FeedforwardNormalization::None, 1.0));
        let stats = network.feedforward_sparsity_stats(1e-3);

        assert_eq!(stats.total_slots, 2);
        assert_eq!(stats.present_synapses, 2);
        assert_eq!(stats.near_zero_synapses, 1);
        assert_eq!(stats.nonzero_synapses(), 1);
        assert!((stats.nonzero_ratio() - 0.5).abs() < 1e-6);
        assert!((stats.near_zero_present_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mnist_network_checkpoint_roundtrip_preserves_state() {
        let network = new_mnist_network(
            4,
            0.2,
            17.0,
            Some(10.4),
            MnistNeuronConfig::default(),
            MnistStdpConfig::default(),
            homeostasis_config(FeedforwardNormalization::L1, 1.0),
        );

        let checkpoint = network.to_checkpoint();
        let restored = PoissonInputNetwork::from_checkpoint(checkpoint.clone());

        assert_eq!(restored.to_checkpoint(), checkpoint);
    }

    #[test]
    fn lateral_inhibition_suppresses_competing_neurons() {
        use crate::snn::neurons::Lif;

        // Create 2 exc + 2 inh Lif neurons with conductance-based dynamics
        let exc = |v_init: f32| Lif::new(
            -65.0, -65.0, -72.0, 200.0,
            0.0, -100.0, 0.6065, 0.7788,
            20.0, 0.05, 2e7, 10,
        );
        let inh = || Lif::new(
            -60.0, -45.0, -40.0, 20.0,
            0.0, -85.0, 0.6065, 0.7788,
            0.0, 0.0, 1e10, 4,
        );

        let mut n0 = exc(-105.0); n0.v = -100.0; // e_inh
        let mut n1 = exc(-105.0); n1.v = -100.0;
        let mut n2 = inh(); n2.v = -85.0; // e_inh
        let mut n3 = inh(); n3.v = -85.0;
        let neurons = vec![n0, n1, n2, n3];

        // Feedforward: 2 inputs -> 2 exc (each column sums to 78)
        let feedforward = SynapseGroup {
            weights: Weight::new_dense(2, 2, vec![78.0, 0.0, 0.0, 78.0]),
            max_weight: 1.0,
            min_weight: 0.0,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        // E->I: OneToOne, weight 10.4
        let e_to_i = SynapseGroup {
            weights: Weight::new_one_to_one(vec![10.4; 2]),
            max_weight: 10.4, min_weight: 10.4,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        // I->E: Dense 2x2, weight 17 off-diagonal, 0 diagonal
        let i_to_e = SynapseGroup {
            weights: Weight::new_dense(2, 2, vec![0.0, 17.0, 17.0, 0.0]),
            max_weight: 17.0, min_weight: 17.0,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };

        let mut net: Network<Lif> = Network::new_from(
            2, 2, neurons, feedforward, e_to_i, i_to_e,
            homeostasis_config(FeedforwardNormalization::None, 78.0),
            vec![0u8; 4], 0,
        );

        // Drive only input 0 for 200 ticks.
        // Neuron 0 should fire, neuron 1 should be suppressed by lateral inhibition.
        let inputs_drive = [true, false];
        let biases = [0.0, 0.0];
        let mut ft = [0usize; 2];

        for _ in 0..200 {
            net.tick(&inputs_drive, &biases, false, Some(&mut ft));
        }

        println!("Neuron 0 spikes: {}, Neuron 1 spikes: {}", ft[0], ft[1]);
        assert!(ft[0] > 0, "Driven neuron should fire");
        // Neuron 1 gets no feedforward input, so should fire much less
        assert!(ft[0] > ft[1] * 5, "Inhibition should strongly suppress undriven neuron: n0={} n1={}", ft[0], ft[1]);
    }

    #[test]
    fn lateral_inhibition_works_with_uniform_drive() {
        use crate::snn::neurons::Lif;

        // Create 10 exc + 10 inh neurons with all inputs driving all neurons equally
        let n = 10;
        let n_in = 4;

        let make_exc = || {
            let mut neuron = Lif::new(
                -65.0, -65.0, -72.0, 200.0,
                0.0, -100.0, 0.6065, 0.7788,
                20.0, 0.05, 2e7, 10,
            );
            neuron.v = -100.0; // e_inh
            neuron
        };
        let make_inh = || {
            let mut neuron = Lif::new(
                -60.0, -45.0, -40.0, 20.0,
                0.0, -85.0, 0.6065, 0.7788,
                0.0, 0.0, 1e10, 4,
            );
            neuron.v = -85.0; // e_inh
            neuron
        };

        let mut neurons = Vec::new();
        for _ in 0..n { neurons.push(make_exc()); }
        for _ in 0..n { neurons.push(make_inh()); }

        let w_ff = 78.0 / n_in as f32;
        let ff_data = vec![w_ff; n_in * n];
        let feedforward = SynapseGroup {
            weights: Weight::new_dense(n_in, n, ff_data),
            max_weight: 1.0, min_weight: 0.0,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        let e_to_i = SynapseGroup {
            weights: Weight::new_one_to_one(vec![10.4; n]),
            max_weight: 10.4, min_weight: 10.4,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };
        let mut i_to_e_data = vec![17.0; n * n];
        for i in 0..n { i_to_e_data[i * n + i] = 0.0; }
        let i_to_e = SynapseGroup {
            weights: Weight::new_dense(n, n, i_to_e_data),
            max_weight: 17.0, min_weight: 17.0,
            lr_plus: 0.0, lr_minus: 0.0,
            tau_plus: 1.0, tau_minus: 1.0,
            tau_post2: 0.0,
            plastic: false,
        };

        // Use random delays (0-20 ticks) to break synchrony, matching the reference
        let max_delay = 20usize;
        let delays: Vec<u8> = (0..n_in * n)
            .map(|_| (rand::random::<f32>() * (max_delay as f32 + 1.0)) as u8)
            .map(|d| d.min(max_delay as u8))
            .collect();

        let mut net: Network<Lif> = Network::new_from(
            n_in, n, neurons, feedforward, e_to_i, i_to_e,
            homeostasis_config(FeedforwardNormalization::None, 78.0),
            delays, max_delay,
        );

        let inputs = vec![true; n_in];
        let biases = vec![0.0; n];
        let mut ft = vec![0usize; n];

        let mut total_inh_spikes = 0usize;

        for tick in 0..700 {
            net.tick(&inputs, &biases, false, Some(&mut ft));

            for i in 0..n {
                if net.buf_fired_neurons[n + i] {
                    total_inh_spikes += 1;
                }
            }

            if tick < 5 || tick % 100 == 0 {
                let exc_fired: usize = (0..n).filter(|&i| net.buf_fired_neurons[i]).count();
                let inh_fired: usize = (0..n).filter(|&i| net.buf_fired_neurons[n + i]).count();
                let gi_sum: f32 = net.buf_exc_gi[..n].iter().sum();
                println!(
                    "t={:3}: exc_fired={} inh_fired={} gi_sum={:.1} n0.v={:.2} n0.ge={:.2} n0.gi={:.2}",
                    tick, exc_fired, inh_fired, gi_sum,
                    net.neurons[0].v, net.neurons[0].ge, net.neurons[0].gi,
                );
            }
        }

        let total_exc_spikes: usize = ft.iter().sum();
        println!("\nTotal exc_spikes={} inh_spikes={}", total_exc_spikes, total_inh_spikes);
        for (i, &f) in ft.iter().enumerate() {
            println!("  Neuron {}: {} spikes", i, f);
        }

        assert!(total_inh_spikes > 0, "Inhibitory neurons never fired!");
        // With uniform drive (all inputs constant, all column sums equal),
        // all neurons converge to the same steady-state rate regardless of delays.
        // The real desynchronization happens with stochastic Poisson input + different random weights.
        // Here we just verify the inhibitory pathway is active.
        assert!(total_exc_spikes > 0, "No excitatory spikes at all");
    }

    /// Diagnostic: trace a sync case tick-by-tick.
    #[test]
    #[ignore] // Intentionally finds pathological gi accumulation that triggers CFL
    fn diagnose_sync_trace() {
        for _attempt in 0..50 {
            let mut net = new_mnist_network(
                400, 1.0, 17.0, Some(10.4),
                MnistNeuronConfig::default(),
                MnistStdpConfig::default(),
                homeostasis_config(FeedforwardNormalization::L1, 78.0),
            );

            let mut rates = vec![0.0f32; 784];
            for i in 0..230 {
                let idx = (6 * 79 + i * 3) % 784;
                rates[idx] = 0.032;
            }

            let biases = vec![0.0f32; 400];
            let mut ft = vec![0usize; 400];

            net.normalize_all_feedforward_weights();
            ft.fill(0);

            // Store per-tick firing events
            let mut events: Vec<(usize, Vec<usize>, usize)> = Vec::new();

            for tick in 0..700 {
                let inputs: Vec<bool> = rates.iter().map(|&r| rand::random::<f32>() < r).collect();
                net.network.tick(&inputs, &biases, false, Some(&mut ft));

                let exc_fired: Vec<usize> = (0..400).filter(|&i| net.network.buf_fired_neurons[i]).collect();
                let inh_fired: usize = (0..400).filter(|&i| net.network.buf_fired_neurons[400 + i]).count();

                if !exc_fired.is_empty() || inh_fired > 0 {
                    events.push((tick, exc_fired, inh_fired));
                }
            }

            let total: usize = ft.iter().sum();
            let firing = ft.iter().filter(|&&f| f > 0).count();
            let max_spikes = *ft.iter().max().unwrap();
            let tied = ft.iter().filter(|&&f| f == max_spikes).count();

            if tied > 3 && max_spikes > 40 {
                println!("SYNC: total={} firing={} max={} tied={}", total, firing, max_spikes, tied);

                for (tick, exc, inh) in events.iter().take(50) {
                    let exc_str: String = if exc.len() <= 10 {
                        format!("{:?}", exc)
                    } else {
                        format!("[{} neurons]", exc.len())
                    };
                    println!("  t={:3}: exc={} inh={}", tick, exc_str, inh);
                }

                let sn = ft.iter().enumerate().find(|(_, f)| **f == max_spikes).unwrap().0;
                println!("\nSync neuron n{}: ge={:.3} gi={:.3} v={:.2} ref={}",
                    sn, net.network.neurons[sn].ge, net.network.neurons[sn].gi,
                    net.network.neurons[sn].v, net.network.neurons[sn].refractory_remaining);
                break;
            }

            if _attempt % 10 == 0 {
                println!("attempt {} => total={} firing={} max={} tied={}", _attempt, total, firing, max_spikes, tied);
            }
        }
    }
}
