#pragma once
#include <algorithm>
#include <cmath>

namespace cpu_brunel {
struct NeuronParameters {
  float p11{}, p21{}, p22{}, p31{}, p32{}, p33{};
  float epsc_initial{}, external_weight{}, external_mean{};
  float threshold{20}, reset{0};
  long refractory_steps{5};
};
struct NeuronState {
  float V{}, Iex{}, dIex{}, Iin{}, dIin{};
  long refractory{}, spike_count{};
  bool step(const NeuronParameters& p, float ex, float in, unsigned int external) {
    if (refractory == 0)
      V = p.p31*dIex + p.p32*Iex + p.p31*dIin + p.p32*Iin + p.p33*V;
    else --refractory;
    Iex = p.p21*dIex + p.p22*Iex;
    dIex *= p.p11;
    Iin = p.p21*dIin + p.p22*Iin;
    dIin *= p.p11;
    dIex += ex;
    dIin += in;
    dIex += p.epsc_initial*p.external_weight*static_cast<float>(external);
    return refractory == 0 && V >= p.threshold;
  }
  void spike(const NeuronParameters& p) {
    V = p.reset; refractory = p.refractory_steps; ++spike_count;
  }
};
struct LearningParameters {
  float learning_rate{0.01f}, depression_ratio{1.05f}, mu{0.4f}, weight_max{100};
  float tau_plus{20}, tau_minus{30};
  bool additive{true};
};
struct LearningState {
  float weight{1}, pre{}, post{}, last_time{};
  void decay(float t, const LearningParameters& p) {
    pre *= std::exp(-(t-last_time)/p.tau_plus);
    post *= std::exp(-(t-last_time)/p.tau_minus);
    last_time = t;
  }
  void on_post(float t, const LearningParameters& p) {
    decay(t,p);
    if (p.additive)
      weight = std::clamp(weight + p.learning_rate*p.weight_max*pre,0.0f,p.weight_max);
    else weight += p.learning_rate*std::pow(weight,p.mu)*pre;
    post += 1.0f;
  }
  // A simultaneous post was processed first. Potentiation used the old pre
  // trace; depression excludes the simultaneous post increment, as in GeNN's
  // nest_causal_boundary branch. The updated weight is delivered.
  float on_pre(float t, bool simultaneous_post, const LearningParameters& p) {
    decay(t,p);
    float effective = post - (simultaneous_post ? 1.0f : 0.0f);
    if (p.additive)
      weight = std::clamp(weight-p.depression_ratio*p.learning_rate*p.weight_max*effective,0.0f,p.weight_max);
    else weight = std::max(0.0f,weight-p.learning_rate*p.depression_ratio*weight*effective);
    pre += 1.0f;
    return weight;
  }
};
}
