use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::snn::neurons::Lif;
use crate::snn::synapse::STDPSynapse;

use super::neurons::Neuron;
use super::synapse::Synapse;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum SpikeSrc {
    Input(usize),
    Neuron(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedforwardNormalization {
    None,
    L1,
    L2,
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

pub struct Network<N: Neuron, S: Synapse> {
    num_inputs: usize,

    neurons: Vec<N>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,
    synapse_forward: BTreeMap<(SpikeSrc, usize), Rc<RefCell<S>>>,
    synapse_reverse: BTreeMap<(usize, SpikeSrc), Rc<RefCell<S>>>,
    feedforward_normalization: FeedforwardNormalization,
}

#[derive(Clone, Copy)]
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

impl<N: Neuron, S: Synapse> Network<N, S> {
    fn renormalize_postsynaptic_feedforward_weights(&mut self, post: usize) {
        if self.feedforward_normalization == FeedforwardNormalization::None {
            return;
        }

        let incoming_synapses: Vec<_> = self
            .synapse_reverse
            .range((post, SpikeSrc::Input(0))..=(post, SpikeSrc::Input(usize::MAX)))
            .map(|((p, src), synapse)| {
                assert!(p == &post);
                assert!(matches!(src, SpikeSrc::Input(_)));
                Rc::clone(synapse)
            })
            .collect();

        if incoming_synapses.is_empty() {
            return;
        }

        let norm_power_sum: f32 = incoming_synapses
            .iter()
            .map(|synapse| self.feedforward_normalization.contribution(synapse.borrow().weight()))
            .sum();
        let scale = self.feedforward_normalization.scale(norm_power_sum);

        for synapse in incoming_synapses {
            let normalized_weight = {
                let synapse = synapse.borrow();
                synapse.weight() / scale
            };
            synapse.borrow_mut().set_weight(normalized_weight);
        }
    }

    pub fn tick(&mut self, inputs: &[bool], biases: &[f32], plasticity_enabled: bool, mut fire_tracker: Option<&mut [usize]>) {
        assert!(inputs.len() == self.num_inputs);
        assert!(biases.len() == self.neurons.len());

        // Collect inputs from last tick
        let fired_inputs: Vec<usize> = inputs.iter().enumerate().filter_map(|(i, fired)| fired.then_some(i)).collect();
        let previously_fired_neurons: Vec<usize> = (0..self.neurons.len())
            .filter(|&i| self.neuron_trackers[i].last_fire == 0)
            .collect();

        let mut feedforward_inputs = vec![0f32; self.neurons.len()];
        let mut recurrent_inputs = vec![0f32; self.neurons.len()];

        for src in fired_inputs.iter().copied() {
            for ((s, post), synapse) in self.synapse_forward.range((SpikeSrc::Input(src), 0)..=(SpikeSrc::Input(src), usize::MAX)) {
                assert!(s == &SpikeSrc::Input(src));
                feedforward_inputs[*post] += synapse.borrow().weight();
            }
        }

        for src in previously_fired_neurons.iter().copied() {
            for ((s, post), synapse) in self.synapse_forward.range((SpikeSrc::Neuron(src), 0)..=(SpikeSrc::Neuron(src), usize::MAX)) {
                assert!(s == &SpikeSrc::Neuron(src));
                recurrent_inputs[*post] += synapse.borrow().weight();
            }
        }

        let mut fired_neurons_now = vec![false; self.neurons.len()];
        for i in 0..self.neurons.len() {
            let fired = self.neurons[i].tick(feedforward_inputs[i] + recurrent_inputs[i] + biases[i], plasticity_enabled);
            fired_neurons_now[i] = fired;
            if let Some(ref mut ft) = fire_tracker {
                if fired {
                    ft[i] += 1;
                }
            }
        }

        if plasticity_enabled {
            let mut touched_feedforward_posts = vec![false; self.neurons.len()];

            // Presynaptic spikes only see postsynaptic spikes that happened before this tick.
            for i in fired_inputs.iter().copied() {
                for ((s, post), synapse) in self.synapse_forward.range_mut((SpikeSrc::Input(i), 0)..=(SpikeSrc::Input(i), usize::MAX)) {
                    assert!(s == &SpikeSrc::Input(i));
                    let post_elapsed = self.neuron_trackers[*post].last_fire.saturating_add(1);
                    synapse.borrow_mut().on_pre_spike(post_elapsed);
                    touched_feedforward_posts[*post] = true;
                }
            }

            for i in fired_neurons_now.iter().enumerate().filter_map(|(i, fired)| fired.then_some(i)) {
                for ((s, post), synapse) in self.synapse_forward.range_mut((SpikeSrc::Neuron(i), 0)..=(SpikeSrc::Neuron(i), usize::MAX)) {
                    assert!(s == &SpikeSrc::Neuron(i));
                    let post_elapsed = self.neuron_trackers[*post].last_fire.saturating_add(1);
                    synapse.borrow_mut().on_pre_spike(post_elapsed);
                }
            }

            // Postsynaptic spikes can see presynaptic spikes from the current input tick.
            for i in fired_neurons_now.iter().enumerate().filter_map(|(i, fired)| fired.then_some(i)) {
                for ((post, s), synapse) in self.synapse_reverse.range((i, SpikeSrc::Input(0))..=(i, SpikeSrc::Neuron(usize::MAX))) {
                    assert!(post == &i);
                    let pre_elapsed = match s {
                        SpikeSrc::Input(idx) => {
                            if inputs[*idx] {
                                0
                            } else {
                                self.input_trackers[*idx].last_fire.saturating_add(1)
                            }
                        }
                        SpikeSrc::Neuron(idx) => {
                            if fired_neurons_now[*idx] {
                                0
                            } else {
                                self.neuron_trackers[*idx].last_fire.saturating_add(1)
                            }
                        }
                    };
                    synapse.borrow_mut().on_post_spike(pre_elapsed);

                    if matches!(s, SpikeSrc::Input(_)) {
                        touched_feedforward_posts[i] = true;
                    }
                }
            }

            if self.feedforward_normalization != FeedforwardNormalization::None {
                for (post, touched) in touched_feedforward_posts.into_iter().enumerate() {
                    if touched {
                        self.renormalize_postsynaptic_feedforward_weights(post);
                    }
                }
            }
        }

        // Update trackers after all learning for the current tick has been computed.
        for (t, &input) in self.input_trackers.iter_mut().zip(inputs.iter()) {
            t.tick(input);
        }

        for i in 0..self.neurons.len() {
            self.neuron_trackers[i].tick(fired_neurons_now[i]);
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

impl<N: Neuron, S: Synapse> Network<N, S> {
    fn new_from(
        num_inputs: usize,
        neurons: Vec<N>,
        synapses: BTreeMap<(SpikeSrc, usize), S>,
        feedforward_normalization: FeedforwardNormalization,
    ) -> Self {
        // Verify synapses
        for ((src, post), _) in synapses.iter() {
            match src {
                SpikeSrc::Input(i) => assert!(*i < num_inputs),
                SpikeSrc::Neuron(i) => assert!(*i < neurons.len()),
            }
            assert!(*post < neurons.len());
        }

        let input_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; num_inputs];
        let neuron_trackers = vec![Tracker { last_fire: usize::MAX >> 1 }; neurons.len()];
        let synapse_forward: BTreeMap<_, _> = synapses.into_iter().map(|(k, v)| (k, Rc::new(RefCell::new(v)))).collect();
        let synapse_reverse = synapse_forward.iter().map(|((s, p), v)| ( (*p, *s), Rc::clone(v) )).collect();
        let mut network = Network {
            num_inputs,
            neurons,
            input_trackers,
            neuron_trackers,
            synapse_forward,
            synapse_reverse,
            feedforward_normalization,
        };

        if feedforward_normalization != FeedforwardNormalization::None {
            for post in 0..network.neurons.len() {
                network.renormalize_postsynaptic_feedforward_weights(post);
            }
        }

        network
    }
}

pub struct PoissonInputNetwork<N: Neuron, S: Synapse> {
    network: Network<N, S>,
}

impl<N: Neuron, S: Synapse> PoissonInputNetwork<N, S> {
    pub fn tick(&mut self, rates: &[f32], biases: &[f32], plasticity_enabled: bool, fire_tracker: Option<&mut [usize]>) {
        let inputs: Vec<bool> = rates.iter().map(|&r| rand::random::<f32>() < r).collect();
        self.network.tick(&inputs, biases, plasticity_enabled, fire_tracker);
    }

    pub fn reset_neurons(&mut self) {
        self.network.reset_neurons();
    }
}

impl<N: Neuron, S: Synapse> From<Network<N, S>> for PoissonInputNetwork<N, S> {
    fn from(network: Network<N, S>) -> Self {
        PoissonInputNetwork { network }
    }
}

// New mnist network, with N real neurons acting as output activation,
// and 784 input poisson encoded neurons
// real neurons have a lateral N x N inhibitory connection, which is not affected by STDP
// The connection rate of 784 x N is given by connection_rate
const MNIST_TAU: f32 = 50.0;
const MNIST_THRESHOLD: f32 = 1.0;
const MNIST_HOMEO_TAU: f32 = 5.0;
const MNIST_HOMEO_INC: f32 = 2.0;
const MNIST_STDP_LR_PLUS: f32 = 0.01;
const MNIST_STDP_LR_MINUS: f32 = 0.02;
const MNIST_STDP_TAU_PLUS: f32 = 10.0;
const MNIST_STDP_TAU_MINUS: f32 = 10.0;
const MNIST_SYNAPSE_WEIGHT: f32 = 0.5;

pub fn new_mnist_network(
    output_num: usize,
    connection_rate: f32,
    inbihitory_weight: f32,
    feedforward_normalization: FeedforwardNormalization,
) -> PoissonInputNetwork<Lif, STDPSynapse> {
    let mut neurons = Vec::with_capacity(output_num);
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(MNIST_THRESHOLD, MNIST_TAU, MNIST_HOMEO_INC, MNIST_HOMEO_TAU));
    }

    let mut synapses = BTreeMap::new();
    let feedforward_max_weight = match feedforward_normalization {
        FeedforwardNormalization::None => MNIST_SYNAPSE_WEIGHT,
        FeedforwardNormalization::L1 | FeedforwardNormalization::L2 => 1.0,
    };

    // Connect inputs
    for i in 0..784 {
        for j in 0..output_num {
            if rand::random::<f32>() < connection_rate {
                synapses.insert((SpikeSrc::Input(i), j), STDPSynapse::new_rand(
                    feedforward_max_weight, 0.0,
                    MNIST_STDP_LR_PLUS, MNIST_STDP_LR_MINUS,
                    MNIST_STDP_TAU_PLUS, MNIST_STDP_TAU_MINUS
                ));
            }
        }
    }

    // Connect lateral inhibitory synapses
    for i in 0..output_num {
        for j in 0..output_num {
            if i != j {
                synapses.insert((SpikeSrc::Neuron(i), j), STDPSynapse::new_rand(
                    inbihitory_weight, inbihitory_weight,
                    0.0, 0.0,
                    1.0, 1.0
                ));
            }
        }
    }

    let network = Network::new_from(784, neurons, synapses, feedforward_normalization);
    network.into()
}

#[cfg(test)]
mod tests {
    use super::{FeedforwardNormalization, Network, SpikeSrc};
    use crate::snn::neurons::Neuron;
    use crate::snn::synapse::Synapse;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct SilentNeuron;

    impl Neuron for SilentNeuron {
        fn tick(&mut self, _input: f32, _plasticity_enabled: bool) -> bool {
            false
        }

        fn reset(&mut self) {}
    }

    struct TestSynapse {
        weight: f32,
    }

    impl Synapse for TestSynapse {
        fn weight(&self) -> f32 {
            self.weight
        }

        fn set_weight(&mut self, weight: f32) {
            self.weight = weight;
        }

        fn on_pre_spike(&mut self, _post_elapsed: usize) {}

        fn on_post_spike(&mut self, _pre_elapsed: usize) {}
    }

    #[test]
    fn new_from_renormalizes_l1_feedforward_weights_to_unit_norm() {
        let mut synapses = BTreeMap::new();
        synapses.insert((SpikeSrc::Input(0), 0), TestSynapse { weight: 0.2 });
        synapses.insert((SpikeSrc::Input(1), 0), TestSynapse { weight: 0.3 });

        let network = Network::new_from(2, vec![SilentNeuron], synapses, FeedforwardNormalization::L1);
        let weights: Vec<f32> = network
            .synapse_reverse
            .range((0, SpikeSrc::Input(0))..=(0, SpikeSrc::Input(usize::MAX)))
            .map(|(_, synapse)| synapse.borrow().weight())
            .collect();

        let l1_norm: f32 = weights.iter().sum();
        assert!((l1_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn new_from_renormalizes_l2_feedforward_weights_to_unit_norm() {
        let mut synapses = BTreeMap::new();
        synapses.insert((SpikeSrc::Input(0), 0), TestSynapse { weight: 0.2 });
        synapses.insert((SpikeSrc::Input(1), 0), TestSynapse { weight: 0.3 });

        let network = Network::new_from(2, vec![SilentNeuron], synapses, FeedforwardNormalization::L2);
        let weights: Vec<f32> = network
            .synapse_reverse
            .range((0, SpikeSrc::Input(0))..=(0, SpikeSrc::Input(usize::MAX)))
            .map(|(_, synapse)| synapse.borrow().weight())
            .collect();

        let l2_norm = weights.iter().map(|weight| weight * weight).sum::<f32>().sqrt();
        assert!((l2_norm - 1.0).abs() < 1e-6);
    }
}
