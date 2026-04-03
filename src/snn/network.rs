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

pub struct Network<N: Neuron, S: Synapse> {
    num_inputs: usize,

    neurons: Vec<N>,
    input_trackers: Vec<Tracker>,
    neuron_trackers: Vec<Tracker>,
    synapse_forward: BTreeMap<(SpikeSrc, usize), Rc<RefCell<S>>>,
    synapse_reverse: BTreeMap<(usize, SpikeSrc), Rc<RefCell<S>>>,
}

#[derive(Clone, Copy)]
pub struct Tracker {
    pub last_fire: usize,
}

impl Tracker {
    fn tick(&mut self, fired: bool) {
        if fired {
            self.last_fire = 0;
        } else {
            self.last_fire += 1;
        }
    }
}

impl<N: Neuron, S: Synapse> Network<N, S> {
    pub fn tick(&mut self, inputs: &[bool], biases: &[f32], update_synapses: bool, mut fire_tracker: Option<&mut [usize]>) {
        assert!(inputs.len() == self.num_inputs);
        assert!(biases.len() == self.neurons.len());

        // Collect inputs from last tick
        let mut neuron_inputs = vec![0f32; self.neurons.len()];
        // Iterate through presynapse sources
        let fired_inputs = inputs.iter().enumerate().filter(|(_, fired)| **fired).map(|(i, _)| i);
        let fired_neurons = (0..self.neurons.len()).filter(|&i| self.neuron_trackers[i].last_fire == 0);
        for src in fired_inputs.clone().map(SpikeSrc::Input).chain(fired_neurons.clone().map(SpikeSrc::Neuron)) {
            // Iterate through synapses from that source
            for ((s, post), synapse) in self.synapse_forward.range((src.clone(), 0)..=(src.clone(), usize::MAX)) {
                assert!(s == &src);
                neuron_inputs[*post] += synapse.borrow().weight();
            }
        }

        if update_synapses {
            // Update synapses, only iterate through synapses with fired pre or post neurons
            
            // Presynaptic input fires
            for i in fired_inputs {
                for ((s, post), synapse) in self.synapse_forward.range_mut((SpikeSrc::Input(i), 0)..=(SpikeSrc::Input(i), usize::MAX)) {
                    assert!(s == &SpikeSrc::Input(i));
                    let pre_fired = self.input_trackers[i].last_fire;
                    let post_fired: usize = self.neuron_trackers[*post].last_fire;
                    synapse.borrow_mut().update(pre_fired, post_fired);
                }
            }

            // Presynaptic neuron fires
            for i in fired_neurons.clone() {
                for ((s, post), synapse) in self.synapse_forward.range_mut((SpikeSrc::Neuron(i), 0)..=(SpikeSrc::Neuron(i), usize::MAX)) {
                    assert!(s == &SpikeSrc::Neuron(i));
                    let pre_fired = self.neuron_trackers[i].last_fire;
                    let post_fired: usize = self.neuron_trackers[*post].last_fire;
                    synapse.borrow_mut().update(pre_fired, post_fired);
                }
            }

            // Postsynaptic neuron fires
            for i in fired_neurons {
                for ((post, s), synapse) in self.synapse_reverse.range((i, SpikeSrc::Input(0))..=(i, SpikeSrc::Neuron(usize::MAX))) {
                    assert!(post == &i);
                    let pre_fired = match s {
                        SpikeSrc::Input(idx) => self.input_trackers[*idx].last_fire,
                        SpikeSrc::Neuron(idx) => self.neuron_trackers[*idx].last_fire,
                    };
                    let post_fired: usize = self.neuron_trackers[i].last_fire;
                    synapse.borrow_mut().update(pre_fired, post_fired);
                }
            }
        }

        // Update input trackers
        for (t, &input) in self.input_trackers.iter_mut().zip(inputs.iter()) {
            t.tick(input);
        }

        // Tick neurons, update neuron trackers
        for i in 0..self.neurons.len() {
            let fired = self.neurons[i].tick(neuron_inputs[i] + biases[i]);
            self.neuron_trackers[i].tick(fired);
            if let Some(ref mut ft) = fire_tracker {
                if fired {
                    ft[i] += 1;
                }
            }
        }
    }

    pub fn reset_neurons(&mut self) {
        for t in self.neurons.iter_mut() {
            t.reset();
        }
    }
}

impl<N: Neuron, S: Synapse> Network<N, S> {
    fn new_from(num_inputs: usize, neurons: Vec<N>, synapses: BTreeMap<(SpikeSrc, usize), S>) -> Self {
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
        Network { num_inputs, neurons, input_trackers, neuron_trackers, synapse_forward, synapse_reverse }
    }
}

pub struct PoissonInputNetwork<N: Neuron, S: Synapse> {
    network: Network<N, S>,
}

impl<N: Neuron, S: Synapse> PoissonInputNetwork<N, S> {
    pub fn tick(&mut self, rates: &[f32], biases: &[f32], update_synapses: bool, fire_tracker: Option<&mut [usize]>) {
        let inputs: Vec<bool> = rates.iter().map(|&r| rand::random::<f32>() < r).collect();
        self.network.tick(&inputs, biases, update_synapses, fire_tracker);
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

pub fn new_mnist_network(output_num: usize, connection_rate: f32, inbihitory_weight: f32) -> PoissonInputNetwork<Lif, STDPSynapse> {
    let mut neurons = Vec::with_capacity(output_num);
    for _ in 0..output_num {
        neurons.push(Lif::new_rand(MNIST_THRESHOLD, MNIST_TAU, MNIST_HOMEO_INC, MNIST_HOMEO_TAU));
    }

    let mut synapses = BTreeMap::new();
    // Connect inputs
    for i in 0..784 {
        for j in 0..output_num {
            if rand::random::<f32>() < connection_rate {
                synapses.insert((SpikeSrc::Input(i), j), STDPSynapse::new_rand(
                    MNIST_SYNAPSE_WEIGHT, 0.0,
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

    let network = Network::new_from(784, neurons, synapses);
    network.into()
}
