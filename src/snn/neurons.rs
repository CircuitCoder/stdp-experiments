use serde::{Deserialize, Serialize};

pub trait Neuron {
    // Tick an neuron, returning true if it fired
    fn tick(&mut self, input: f32, plasticity_enabled: bool) -> bool;
    fn reset(&mut self);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lif {
    v: f32,
    rest_potential: f32,
    reset_potential: f32,
    base_threshold: f32,
    active_threshold: f32,
    tau: f32,
    homeo_tau: f32,
    homeo_inc: f32,
    refractory_ticks: usize,
    refractory_remaining: usize,
}

impl Lif {
    pub fn new(
        rest_potential: f32,
        reset_potential: f32,
        threshold: f32,
        tau: f32,
        homeo_inc: f32,
        homeo_tau: f32,
        refractory_ticks: usize,
    ) -> Self {
        assert!(tau > 0.0);
        assert!(homeo_tau > 0.0);
        assert!(threshold > rest_potential);
        assert!(threshold > reset_potential);

        Lif {
            v: rest_potential,
            rest_potential,
            reset_potential,
            base_threshold: threshold,
            active_threshold: threshold,
            tau,
            homeo_tau,
            homeo_inc,
            refractory_ticks,
            refractory_remaining: 0,
        }
    }

    pub fn new_rand(
        rest_potential: f32,
        reset_potential: f32,
        threshold: f32,
        tau: f32,
        homeo_inc: f32,
        homeo_tau: f32,
        refractory_ticks: usize,
    ) -> Self {
        let threshold_gap = (threshold - rest_potential).max(f32::EPSILON);
        let random_unit = rand::random::<f32>().clamp(f32::EPSILON, 1.0 - f32::EPSILON);
        let mut neuron = Self::new(
            rest_potential,
            reset_potential,
            threshold,
            tau,
            homeo_inc,
            homeo_tau,
            refractory_ticks,
        );
        neuron.v = rest_potential - threshold_gap * random_unit.ln();
        neuron
    }
}

impl Neuron for Lif {
    fn tick(&mut self, input: f32, plasticity_enabled: bool) -> bool {
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            self.v = self.reset_potential;
            if plasticity_enabled {
                self.active_threshold += (self.base_threshold - self.active_threshold) / self.homeo_tau;
            }
            return false;
        }

        self.v += input;
        if self.v >= self.active_threshold {
            self.v = self.reset_potential;
            self.refractory_remaining = self.refractory_ticks;
            if plasticity_enabled {
                self.active_threshold += self.homeo_inc;
            }
            true
        } else {
            self.v += (self.rest_potential - self.v) / self.tau;
            if plasticity_enabled {
                self.active_threshold += (self.base_threshold - self.active_threshold) / self.homeo_tau;
            }
            false
        }
    }

    fn reset(&mut self) {
        self.v = self.rest_potential;
        self.refractory_remaining = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{Lif, Neuron};

    #[test]
    fn refractory_ticks_block_spikes_until_their_budget_is_spent() {
        let mut neuron = Lif::new(0.0, 0.0, 1.0, 10.0, 0.0, 1.0, 2);

        assert!(neuron.tick(2.0, false));
        assert!(!neuron.tick(2.0, false));
        assert!(!neuron.tick(2.0, false));
        assert!(neuron.tick(2.0, false));
    }
}
