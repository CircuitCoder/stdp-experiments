use serde::{Deserialize, Serialize};

pub trait Neuron {
    // Tick a neuron, returning true if it fired.
    // `input` is a direct current injection (e.g. bias/noise).
    fn tick(&mut self, input: f32, plasticity_enabled: bool) -> bool;
    fn reset(&mut self);
    // Add excitatory / inhibitory conductance (called by Network before tick)
    fn add_ge(&mut self, ge: f32);
    fn add_gi(&mut self, gi: f32);
}

/// Conductance-based LIF neuron (FTDT discretization).
///
/// Each tick the network adds pre-synaptic weights to `ge` / `gi`,
/// then calls `tick()`.  The voltage ODE
///
///   dv/dt = ((v_rest - v) + ge*(e_exc - v) + gi*(e_inh - v)) / tau + input
///
/// is linear in v, so we solve it analytically (exact exponential integration):
///
///   alpha = (1 + ge + gi) / tau
///   v_inf = (v_rest + ge*e_exc + gi*e_inh + input*tau) / (1 + ge + gi)
///   v(t+dt) = v_inf + (v - v_inf) * exp(-alpha)
///
/// This is unconditionally stable regardless of conductance magnitude.
/// After the step, conductances decay:
///   ge *= ge_decay;   gi *= gi_decay;
///
/// `ge_decay` / `gi_decay` are pre-computed as `exp(-1/tau_ge)` etc.
///
/// Threshold adaptation (theta):
///   On spike: theta += theta_plus
///   Each tick: theta *= exp(-1/theta_tc)   (exact exponential decay, f64 precision)
///   Effective threshold = v_thresh + theta
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lif {
    pub v: f32,
    pub rest_potential: f32,
    pub reset_potential: f32,
    pub threshold: f32,
    pub tau: f32,
    // Conductance state — written by the Network before each tick
    pub ge: f32,
    pub gi: f32,
    // Reversal potentials
    pub e_exc: f32,
    pub e_inh: f32,
    // Conductance exponential decay factors (per tick)
    pub ge_decay: f32,
    pub gi_decay: f32,
    // Threshold adaptation (f64 for precision: theta_tc=2e7 makes f32 decay a no-op)
    pub theta: f64,
    pub theta_plus: f64,
    pub theta_decay: f64,  // pre-computed exp(-1/theta_tc)
    // Refractoriness
    pub refractory_ticks: usize,
    pub refractory_remaining: usize,
}

impl Lif {
    pub fn new(
        rest_potential: f32,
        reset_potential: f32,
        threshold: f32,
        tau: f32,
        e_exc: f32,
        e_inh: f32,
        ge_decay: f32,
        gi_decay: f32,
        theta_init: f32,
        theta_plus: f32,
        theta_tc: f32,
        refractory_ticks: usize,
    ) -> Self {
        assert!(tau > 0.0);
        assert!(theta_tc > 0.0);

        Lif {
            v: rest_potential,
            rest_potential,
            reset_potential,
            threshold,
            tau,
            ge: 0.0,
            gi: 0.0,
            e_exc,
            e_inh,
            ge_decay,
            gi_decay,
            theta: theta_init as f64,
            theta_plus: theta_plus as f64,
            theta_decay: (-1.0f64 / theta_tc as f64).exp(),
            refractory_ticks,
            refractory_remaining: 0,
        }
    }

    pub fn new_rand(
        rest_potential: f32,
        reset_potential: f32,
        threshold: f32,
        tau: f32,
        e_exc: f32,
        e_inh: f32,
        ge_decay: f32,
        gi_decay: f32,
        theta_init: f32,
        theta_plus: f32,
        theta_tc: f32,
        refractory_ticks: usize,
    ) -> Self {
        // Match the reference: v = v_rest - 40 mV, well below threshold so
        // the competitive dynamics develop gradually from input drive.
        let mut neuron = Self::new(
            rest_potential, reset_potential, threshold, tau,
            e_exc, e_inh, ge_decay, gi_decay,
            theta_init, theta_plus, theta_tc,
            refractory_ticks,
        );
        neuron.v = (rest_potential - 40.0).max(e_inh);
        neuron
    }

    #[inline(always)]
    pub fn effective_threshold(&self) -> f32 {
        self.threshold + self.theta as f32
    }
}

impl Neuron for Lif {
    fn tick(&mut self, input: f32, plasticity_enabled: bool) -> bool {
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            self.v = self.reset_potential;
            // Reference (Brian v1 freeze=True): equations are NOT applied during
            // refractory.  ge/gi don't decay, theta doesn't decay, but input
            // from add_ge/add_gi still accumulates.  This lets the "winner"
            // neuron exit refractory with high ge and re-fire quickly.
            return false;
        }

        // Theta decays every non-refractory tick (exact exponential, f64 precision)
        if plasticity_enabled {
            self.theta *= self.theta_decay;
        }

        // Exact exponential integration of the linear voltage ODE (dt = 1 tick).
        //
        //   dv/dt = ((v_rest - v) + ge*(e_exc - v) + gi*(e_inh - v)) / tau + input
        //         = -alpha * (v - v_inf)
        //
        // where alpha = (1 + ge + gi) / tau
        //       v_inf = (v_rest + ge*e_exc + gi*e_inh + input*tau) / (1 + ge + gi)
        //
        // Solution: v(t+1) = v_inf + (v(t) - v_inf) * exp(-alpha)
        let gtot = 1.0 + self.ge + self.gi;
        let alpha = gtot / self.tau;
        let v_inf = (self.rest_potential + self.ge * self.e_exc
            + self.gi * self.e_inh + input * self.tau)
            / gtot;
        self.v = v_inf + (self.v - v_inf) * (-alpha).exp();

        // Decay conductances
        self.ge *= self.ge_decay;
        self.gi *= self.gi_decay;

        let threshold = self.effective_threshold();
        if self.v >= threshold {
            self.v = self.reset_potential;
            self.refractory_remaining = self.refractory_ticks;
            if plasticity_enabled {
                self.theta += self.theta_plus;
            }
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.v = self.rest_potential;
        self.ge = 0.0;
        self.gi = 0.0;
        self.refractory_remaining = 0;
    }

    fn add_ge(&mut self, ge: f32) {
        self.ge += ge;
    }

    fn add_gi(&mut self, gi: f32) {
        self.gi += gi;
    }
}

#[cfg(test)]
mod tests {
    use super::{Lif, Neuron};

    #[test]
    fn refractory_ticks_block_spikes_until_their_budget_is_spent() {
        // With conductance-based model: inject via ge with e_exc > threshold.
        // tau=10 keeps the system well within RK4 stability (h*alpha < 2.78).
        let mut neuron = Lif::new(
            0.0,   // rest
            0.0,   // reset
            1.0,   // threshold
            10.0,  // tau
            10.0,  // e_exc (high reversal so ge drives v up)
            -10.0, // e_inh
            0.9,   // ge_decay
            0.9,   // gi_decay
            0.0,   // theta_init
            0.0,   // theta_plus
            1e10,  // theta_tc (effectively no adaptation)
            2,     // refractory_ticks
        );

        // Inject enough conductance to fire (v ≈ 1.65 with these params)
        neuron.ge = 2.0;
        assert!(neuron.tick(0.0, false));
        neuron.ge = 2.0;
        assert!(!neuron.tick(0.0, false)); // refractory
        neuron.ge = 2.0;
        assert!(!neuron.tick(0.0, false)); // refractory
        neuron.ge = 2.0;
        assert!(neuron.tick(0.0, false));  // fires again
    }
}
