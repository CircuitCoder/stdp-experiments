pub trait Neuron {
    // Tick an neuron, returning true if it fired
    fn tick(&mut self, input: f32) -> bool;
    fn reset(&mut self);
}

pub struct Lif {
    v: f32,
    base_threshold: f32,
    active_threshold: f32,
    tau: f32,
    homeo_tau: f32,
    homeo_inc: f32,
}

impl Lif {
    pub fn new_rand(threshold: f32, tau: f32, homeo_inc: f32, homeo_tau: f32) -> Self {
        // Random potential below threshold according to exponential distribution
        // TODO: is this correct?
        let v = -threshold * rand::random::<f32>().ln();
        Lif { v, base_threshold: threshold, active_threshold: threshold, tau, homeo_tau, homeo_inc }
    }
}

impl Neuron for Lif {
    fn tick(&mut self, input: f32) -> bool {
        self.v += input;
        if self.v >= self.active_threshold {
            self.v = 0.0;
            self.active_threshold += self.homeo_inc;
            true
        } else {
            self.v *= 1.0 - 1.0 / self.tau;
            self.v = self.v.max(0.0);
            self.active_threshold += (self.base_threshold - self.active_threshold) / self.homeo_tau;
            false
        }
    }

    fn reset(&mut self) {
        self.v = 0.0;
        // Don't reset active threshold
    }
}
