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
    pub membrane_tau: f32,
    pub threshold: f32,
    pub reset_potential: f32,
    pub refractory_ticks: usize,
    pub threshold_homeostasis_tau: f32,
    pub threshold_homeostasis_inc: f32,
    pub inhibitory_membrane_tau: Option<f32>,
    pub inhibitory_threshold: Option<f32>,
    pub inhibitory_reset_potential: Option<f32>,
    pub inhibitory_refractory_ticks: Option<usize>,
    pub inhibitory_threshold_homeostasis_tau: Option<f32>,
    pub inhibitory_threshold_homeostasis_inc: Option<f32>,
}

impl Default for MnistNeuronConfig {
    fn default() -> Self {
        Self {
            membrane_tau: MNIST_TAU,
            threshold: MNIST_THRESHOLD,
            reset_potential: 0.0,
            refractory_ticks: 0,
            threshold_homeostasis_tau: MNIST_HOMEO_TAU,
            threshold_homeostasis_inc: MNIST_HOMEO_INC,
            inhibitory_membrane_tau: None,
            inhibitory_threshold: None,
            inhibitory_reset_potential: None,
            inhibitory_refractory_ticks: None,
            inhibitory_threshold_homeostasis_tau: None,
            inhibitory_threshold_homeostasis_inc: None,
        }
    }
}

impl MnistNeuronConfig {
    fn inhibitory_membrane_tau(self) -> f32 {
        self.inhibitory_membrane_tau.unwrap_or(self.membrane_tau)
    }

    fn inhibitory_threshold(self) -> f32 {
        self.inhibitory_threshold.unwrap_or(self.threshold)
    }

    fn inhibitory_reset_potential(self) -> f32 {
        self.inhibitory_reset_potential.unwrap_or(self.reset_potential)
    }

    fn inhibitory_refractory_ticks(self) -> usize {
        self.inhibitory_refractory_ticks.unwrap_or(self.refractory_ticks)
    }

    fn inhibitory_threshold_homeostasis_tau(self) -> f32 {
        self.inhibitory_threshold_homeostasis_tau.unwrap_or(self.threshold_homeostasis_tau)
    }

    fn inhibitory_threshold_homeostasis_inc(self) -> f32 {
        self.inhibitory_threshold_homeostasis_inc.unwrap_or(self.threshold_homeostasis_inc)
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

impl FeedforwardNormalization {
    fn contribution(self, weight: f32) -> f32 {
        match self {
            FeedforwardNormalization::None => 0.0,
            FeedforwardNormalization::L1 => weight.abs(),
            FeedforwardNormalization::L2 => weight * weight,
        }
    }

    fn scale(self, norm_power_sum: f32) -> f32 {
        const EPSILON: f32 = 1e-6;

        match self {
            FeedforwardNormalization::None => 1.0,
            FeedforwardNormalization::L1 => norm_power_sum.max(EPSILON),
            FeedforwardNormalization::L2 => norm_power_sum.sqrt().max(EPSILON),
        }
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

    // Reusable per-tick buffers (avoid allocation in hot loop)
    buf_exc_inputs: Vec<f32>,
    buf_inh_inputs: Vec<f32>,
    buf_fired_neurons: Vec<bool>,
    buf_touched_posts: Vec<bool>,
    buf_fired_inputs: Vec<usize>,
}

#[derive(Clone)]
pub struct NetworkRuntimeState<N: Clone> {
    neurons: Vec<N>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,
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

        let normalization = self.feedforward_homeostasis.normalization;
        let power_sum: f32 = self.feedforward.weights.iter_post(post)
            .map(|(_pre, w)| normalization.contribution(w))
            .sum();
        let scale = normalization.scale(power_sum);
        let target_gain = self.feedforward_post_gains[post];
        let min_w = self.feedforward.min_weight;
        let max_w = self.feedforward.max_weight;

        for (_pre, w) in self.feedforward.weights.iter_post_mut(post) {
            *w = (target_gain * *w / scale).clamp(min_w, max_w);
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

        // 2. Accumulate feedforward: Input -> Exc
        self.buf_exc_inputs[..n].fill(0.0);
        for &src in &self.buf_fired_inputs {
            self.feedforward.weights.accumulate_pre(src, &mut self.buf_exc_inputs[..n]);
        }

        // 3. Accumulate e_to_i: Exc -> Inh (from previously fired exc neurons)
        self.buf_inh_inputs[..num_inh].fill(0.0);
        for i in 0..n {
            if self.neuron_trackers[i].last_fire == 0 {
                self.e_to_i.weights.accumulate_pre(i, &mut self.buf_inh_inputs[..num_inh]);
            }
        }

        // 4. Accumulate i_to_e: Inh -> Exc (from previously fired inh neurons)
        for i in 0..num_inh {
            if self.neuron_trackers[n + i].last_fire == 0 {
                self.i_to_e.weights.accumulate_pre(i, &mut self.buf_exc_inputs[..n]);
            }
        }

        // 5. Tick neurons
        self.buf_fired_neurons[..num_neurons].fill(false);
        for i in 0..n {
            let fired = self.neurons[i].tick(self.buf_exc_inputs[i] + biases[i], plasticity_enabled);
            self.buf_fired_neurons[i] = fired;
            if let Some(ref mut ft) = fire_tracker {
                if fired {
                    ft[i] += 1;
                }
            }
        }
        for i in 0..num_inh {
            let fired = self.neurons[n + i].tick(self.buf_inh_inputs[i], plasticity_enabled);
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
                    let post_elapsed = self.neuron_trackers[post].last_fire.saturating_add(1);
                    let delta = -lr_minus * (-(post_elapsed as f32) / tau_minus).exp();
                    *w = (*w + delta).clamp(min_weight, max_weight);
                    self.buf_touched_posts[post] = true;
                }
            }

            // Potentiation: post-spike rule (postsynaptic spikes can see current input spikes)
            for j in 0..n {
                if self.buf_fired_neurons[j] {
                    for (pre, w) in self.feedforward.weights.iter_post_mut(j) {
                        let pre_elapsed = if inputs[pre] {
                            0
                        } else {
                            self.input_trackers[pre].last_fire.saturating_add(1)
                        };
                        let delta = lr_plus * (-(pre_elapsed as f32) / tau_plus).exp();
                        *w = (*w + delta).clamp(min_weight, max_weight);
                        self.buf_touched_posts[j] = true;
                    }
                }
            }

            // Renormalize touched columns
            if self.feedforward_homeostasis.normalization != FeedforwardNormalization::None {
                for post in 0..n {
                    if self.buf_touched_posts[post] {
                        self.renormalize_postsynaptic_feedforward_weights(post);
                    }
                }
            }
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
    }
}

impl<N: Neuron + Clone> Network<N> {
    fn snapshot_runtime_state(&self) -> NetworkRuntimeState<N> {
        NetworkRuntimeState {
            neurons: self.neurons.clone(),
            input_trackers: self.input_trackers.clone(),
            neuron_trackers: self.neuron_trackers.clone(),
        }
    }

    fn restore_runtime_state(&mut self, state: NetworkRuntimeState<N>) {
        self.neurons = state.neurons;
        self.input_trackers = state.input_trackers;
        self.neuron_trackers = state.neuron_trackers;
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

        let input_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; num_inputs];
        let neuron_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; num_neurons];
        let feedforward_post_gains = vec![feedforward_homeostasis.post_renorm_gain; num_output_neurons];
        let feedforward_rate_ema = vec![feedforward_homeostasis.slow_scaling_target_rate; num_output_neurons];

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
            buf_exc_inputs: vec![0.0; num_output_neurons],
            buf_inh_inputs: vec![0.0; num_inh],
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
                buf_exc_inputs: vec![0.0; num_output],
                buf_inh_inputs: vec![0.0; num_inh],
                buf_fired_neurons: vec![false; num_neurons],
                buf_touched_posts: vec![false; num_output],
                buf_fired_inputs: Vec::with_capacity(num_inputs),
            },
            buf_poisson_inputs: Vec::with_capacity(num_inputs),
        }
    }
}

// --- MNIST network construction ---

const MNIST_TAU: f32 = 50.0;
const MNIST_THRESHOLD: f32 = 1.0;
const MNIST_HOMEO_TAU: f32 = 5.0;
const MNIST_HOMEO_INC: f32 = 2.0;
const MNIST_STDP_LR_PLUS: f32 = 0.01;
const MNIST_STDP_LR_MINUS: f32 = 0.02;
const MNIST_STDP_TAU_PLUS: f32 = 10.0;
const MNIST_STDP_TAU_MINUS: f32 = 10.0;
const MNIST_SYNAPSE_WEIGHT: f32 = 0.5;
const MNIST_REF_E_TO_I_WEIGHT: f32 = 10.4;
const MNIST_REF_I_TO_E_WEIGHT: f32 = 17.0;

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
    assert!(neuron_config.threshold > 0.0);
    assert!(neuron_config.reset_potential < neuron_config.threshold);
    assert!(neuron_config.threshold_homeostasis_tau > 0.0);
    assert!(neuron_config.threshold_homeostasis_inc >= 0.0);
    assert!(neuron_config.inhibitory_membrane_tau() > 0.0);
    assert!(neuron_config.inhibitory_threshold() > 0.0);
    assert!(neuron_config.inhibitory_reset_potential() < neuron_config.inhibitory_threshold());
    assert!(neuron_config.inhibitory_threshold_homeostasis_tau() > 0.0);
    assert!(neuron_config.inhibitory_threshold_homeostasis_inc() >= 0.0);
    assert!(inhibitory_weight <= 0.0);
    if let Some(weight) = excitatory_to_inhibitory_weight {
        assert!(weight >= 0.0);
    }

    // Build neurons: N excitatory + N inhibitory
    let mut neurons = Vec::with_capacity(output_num * 2);
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(
            0.0,
            neuron_config.reset_potential,
            neuron_config.threshold,
            neuron_config.membrane_tau,
            neuron_config.threshold_homeostasis_inc,
            neuron_config.threshold_homeostasis_tau,
            neuron_config.refractory_ticks,
        ));
    }
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(
            0.0,
            neuron_config.inhibitory_reset_potential(),
            neuron_config.inhibitory_threshold(),
            neuron_config.inhibitory_membrane_tau(),
            neuron_config.inhibitory_threshold_homeostasis_inc(),
            neuron_config.inhibitory_threshold_homeostasis_tau(),
            neuron_config.inhibitory_refractory_ticks(),
        ));
    }

    let excitatory_to_inhibitory_weight = excitatory_to_inhibitory_weight
        .unwrap_or_else(|| inhibitory_weight.abs() * (MNIST_REF_E_TO_I_WEIGHT / MNIST_REF_I_TO_E_WEIGHT));
    let feedforward_max_weight = stdp_config.resolved_weight_max(feedforward_homeostasis);
    let stdp_lr_plus = stdp_config.resolved_lr_plus();
    let stdp_lr_minus = stdp_config.resolved_lr_minus();
    let stdp_tau_plus = stdp_config.resolved_tau_plus();
    let stdp_tau_minus = stdp_config.resolved_tau_minus();

    // Feedforward: Input(784) -> Exc(N), Dense
    let feedforward_data: Vec<f32> = (0..784 * output_num)
        .map(|_| {
            if rand::random::<f32>() < connection_rate {
                feedforward_max_weight * rand::random::<f32>()
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
        plastic: false,
    };

    let network = Network::new_from(784, output_num, neurons, feedforward, e_to_i, i_to_e, feedforward_homeostasis);
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
            plastic: false,
        }
    }

    /// Build a minimal test network: `num_inputs` inputs, `num_exc` excitatory
    /// SilentNeurons, `num_exc` inhibitory SilentNeurons, with the given
    /// feedforward weights and zero lateral weights.
    fn test_network(
        num_inputs: usize,
        num_exc: usize,
        feedforward_weights: Vec<f32>,
        homeostasis: FeedforwardHomeostasisConfig,
    ) -> Network<SilentNeuron> {
        let feedforward = SynapseGroup {
            weights: Weight::new_dense(num_inputs, num_exc, feedforward_weights),
            max_weight: f32::MAX,
            min_weight: 0.0,
            lr_plus: 0.0,
            lr_minus: 0.0,
            tau_plus: 1.0,
            tau_minus: 1.0,
            plastic: false,
        };
        let e_to_i = dummy_synapse_group(Weight::new_one_to_one(vec![0.0; num_exc]));
        let i_to_e = dummy_synapse_group(Weight::new_dense(num_exc, num_exc, vec![0.0; num_exc * num_exc]));
        let neurons = vec![SilentNeuron; num_exc * 2];

        Network::new_from(num_inputs, num_exc, neurons, feedforward, e_to_i, i_to_e, homeostasis)
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
        let network = test_network(2, 1, vec![0.2, 0.3], homeostasis_config(FeedforwardNormalization::L2, 2.0));
        let weights: Vec<f32> = network.feedforward.weights.iter_post(0).map(|(_pre, w)| w).collect();
        let l2_norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        assert!((l2_norm - 2.0).abs() < 1e-6);
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
            -0.1,
            Some(0.05),
            MnistNeuronConfig::default(),
            MnistStdpConfig::default(),
            homeostasis_config(FeedforwardNormalization::L1, 1.0),
        );

        let checkpoint = network.to_checkpoint();
        let restored = PoissonInputNetwork::from_checkpoint(checkpoint.clone());

        assert_eq!(restored.to_checkpoint(), checkpoint);
    }
}
