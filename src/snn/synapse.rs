pub trait Synapse {
    fn weight(&self) -> f32;
    fn set_weight(&mut self, weight: f32);
    fn on_pre_spike(&mut self, post_elapsed: usize);
    fn on_post_spike(&mut self, pre_elapsed: usize);
}

pub struct STDPSynapse {
    weight: f32,
    max_weight: f32,
    min_weight: f32,

    lr_plus: f32,
    lr_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
}

impl Synapse for STDPSynapse {
    fn weight(&self) -> f32 {
        self.weight
    }

    fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
        self.clamp_weight();
    }

    fn on_pre_spike(&mut self, post_elapsed: usize) {
        let delta_w = -self.lr_minus * (-(post_elapsed as f32) / self.tau_minus).exp();
        self.weight += delta_w;

        self.clamp_weight();
    }

    fn on_post_spike(&mut self, pre_elapsed: usize) {
        let delta_w = self.lr_plus * (-(pre_elapsed as f32) / self.tau_plus).exp();
        self.weight += delta_w;

        self.clamp_weight();
    }
}

impl STDPSynapse {
    fn clamp_weight(&mut self) {
        // Clamp weight
        if self.weight > self.max_weight {
            self.weight = self.max_weight;
        } else if self.weight < self.min_weight {
            self.weight = self.min_weight;
        }
    }

    pub fn new_rand(max_weight: f32, min_weight: f32, lr_plus: f32, lr_minus: f32, tau_plus: f32, tau_minus: f32) -> Self {
        let weight = min_weight + (max_weight - min_weight) * rand::random::<f32>();
        STDPSynapse { weight, max_weight, min_weight, lr_plus, lr_minus, tau_plus, tau_minus }
    }
}

#[cfg(test)]
mod tests {
    use super::{STDPSynapse, Synapse};

    #[test]
    fn post_spike_uses_pre_elapsed_for_potentiation() {
        let mut synapse = STDPSynapse {
            weight: 0.5,
            max_weight: 1.0,
            min_weight: 0.0,
            lr_plus: 0.1,
            lr_minus: 0.1,
            tau_plus: 10.0,
            tau_minus: 10.0,
        };

        synapse.on_post_spike(5);

        let expected = 0.5 + 0.1 * (-0.5f32).exp();
        assert!((synapse.weight() - expected).abs() < 1e-6);
    }

    #[test]
    fn pre_spike_uses_post_elapsed_for_depression() {
        let mut synapse = STDPSynapse {
            weight: 0.5,
            max_weight: 1.0,
            min_weight: 0.0,
            lr_plus: 0.1,
            lr_minus: 0.1,
            tau_plus: 10.0,
            tau_minus: 10.0,
        };

        synapse.on_pre_spike(5);

        let expected = 0.5 - 0.1 * (-0.5f32).exp();
        assert!((synapse.weight() - expected).abs() < 1e-6);
    }
}
