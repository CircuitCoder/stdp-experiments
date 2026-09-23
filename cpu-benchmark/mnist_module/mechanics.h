#pragma once
#include <algorithm>
#include <cmath>

namespace cpu_mnist {
struct Parameters {
  long kind{1}, rule{3}, refractory_steps{10}; // input=0, E=1, I=2
  float tau_m{100}, v_rest{-65}, v_reset{-65}, v_threshold{-52};
  float e_exc{0}, e_inh{-100}, theta_offset{20}, theta_plus{0.05f};
  float ge_half_decay{static_cast<float>(std::exp(-0.25))};
  float gi_half_decay{static_cast<float>(std::exp(-0.125))};
  float ge_decay{static_cast<float>(std::exp(-0.5))};
  float gi_decay{static_cast<float>(std::exp(-0.25))};
  float theta_decay{static_cast<float>(std::exp(-0.5 / 1e7))};
  float pre_tau{20}, post1_tau{20}, post2_tau{40};
  float potentiation{0.01f}, depression{0.0001f}, weight_max{1};
  float pre_target{0.4f}, pre_exponent{0.2f}, post_exponent{0.2f};
  float plasticity{1}, rate_hz{0};
};
struct State {
  float V{-105}, ge{}, gi{}, theta{20}, post_trace{}, last_input_current{};
  long refractory{}, spike_count{}, last_post{-1}, previous_post{-1}, clock_step{};
  bool previous_fired{};
  bool integrate(const Parameters& p, float ex, float in) {
    ge += ex; gi += in;
    if (refractory > 0) --refractory;
    else {
      float ge_mid = ge * p.ge_half_decay, gi_mid = gi * p.gi_half_decay;
      float g_mid = 1.0f + ge_mid + gi_mid;
      float v_inf = (p.v_rest + ge_mid * p.e_exc + gi_mid * p.e_inh) / g_mid;
      // Match GeNN's unqualified exp: float argument, double libm result.
      V = v_inf + (V - v_inf) * std::exp(double(-(0.5f * g_mid) / p.tau_m));
      ge *= p.ge_decay; gi *= p.gi_decay;
      if (p.kind == 1) theta *= p.theta_decay;
    }
    return refractory == 0 && V > (p.kind == 1 ? theta - p.theta_offset + p.v_threshold : p.v_threshold);
  }
  void fire(const Parameters& p, long step) {
    V = p.v_reset; refractory = p.refractory_steps;
    if (p.kind == 1) theta += p.theta_plus;
    previous_post = last_post; last_post = step; ++spike_count;
  }
};
inline float clamp_weight(double value, const Parameters& p) {
  return std::fmin(double(p.weight_max), std::fmax(0.0, value));
}
inline float on_pre(float weight, float old_post_trace, long old_post, long step, const Parameters& p) {
  if (p.plasticity == 0) return weight;
  if (p.rule == 1) return weight;
  if (p.rule == 2)
    return clamp_weight(weight - p.plasticity * p.depression * old_post_trace
      * std::pow(double(weight), double(p.pre_exponent)), p);
  float post1 = old_post >= 0 ? std::exp(double(-float(step - old_post) * 0.5f / p.post1_tau)) : 0.0;
  return clamp_weight(weight - (p.plasticity * p.depression * post1), p);
}
inline float on_post(float weight, float pre_trace, long last_pre, long previous_post,
                     long step, const Parameters& p) {
  if (p.plasticity == 0) return weight;
  if (p.rule == 1) {
    float delta = p.plasticity * p.potentiation * (pre_trace - p.pre_target)
      * std::pow(double(p.weight_max - weight), double(p.post_exponent));
    return clamp_weight(weight + delta, p);
  }
  if (p.rule == 2)
    return clamp_weight(weight + p.plasticity * p.potentiation * (pre_trace - p.pre_target)
      * std::pow(double(p.weight_max - weight), double(p.post_exponent)), p);
  float pre = last_pre >= 0 ? std::exp(double(-float(step - last_pre) * 0.5f / p.pre_tau)) : 0.0;
  float post2 = previous_post >= 0 ? std::exp(double(-float(step - previous_post) * 0.5f / p.post2_tau)) : 0.0;
  return clamp_weight(weight + p.plasticity * p.potentiation * pre * post2, p);
}
}
