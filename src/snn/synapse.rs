pub trait Synapse {
    fn weight(&self) -> f32;
    fn update(&mut self, pre_last_fire: usize, post_last_fire: usize);
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

    fn update(&mut self, pre_last_fire: usize, post_last_fire: usize) {
        if pre_last_fire > 0 && post_last_fire == 0 {
            // Post fired, pre did not, synaptic potentiation
            let delta_w: f32 = self.lr_plus * (- (post_last_fire as f32) / self.tau_plus).exp();
            self.weight += delta_w;
        } else if post_last_fire > 0 && pre_last_fire == 0 {
            // Pre fired, post did not, synaptic depression
            let delta_w = -self.lr_minus * (- (pre_last_fire as f32) / self.tau_minus).exp();
            self.weight += delta_w;
        }

        // Clamp weight
        if self.weight > self.max_weight {
            self.weight = self.max_weight;
        } else if self.weight < self.min_weight {
            self.weight = self.min_weight;
        }
    }
}

impl STDPSynapse {
    pub fn new_rand(max_weight: f32, min_weight: f32, lr_plus: f32, lr_minus: f32, tau_plus: f32, tau_minus: f32) -> Self {
        let weight = min_weight + (max_weight - min_weight) * rand::random::<f32>();
        STDPSynapse { weight, max_weight, min_weight, lr_plus, lr_minus, tau_plus, tau_minus }
    }
}
