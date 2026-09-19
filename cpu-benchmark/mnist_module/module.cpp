#include "mechanics.h"
#include <array>
#include <vector>
#include <cstdint>
#include "archiving_node.h"
#include "dict_util.h"
#include "event.h"
#include "kernel_manager.h"
#include "model_manager_impl.h"
#include "nest_extension_interface.h"
#include "random_generators.h"
#include "ring_buffer.h"

namespace cpu_mnist {
constexpr size_t input_count = 784;
struct InputTrace { float x{}; long last_spike{-1}; };
struct alignas(64) SharedInput { std::array<InputTrace, 2> trace; };
// One network / one process. Each input owns its two slots. All synapses have
// delay == resolution, so NEST barriers separate ticks. At tick s, inputs write
// slot s%2 and E nodes read slot (s-1)%2; no concurrent slot has two writers or
// a reader and writer. This shares the per-input trace as GeNN does.
std::array<SharedInput, input_count> inputs;

inline void read_float(const Dictionary& d, const std::string& key, float& value) {
  double v = value; d.update_value(key, v); value = static_cast<float>(v);
}

class Neuron : public nest::ArchivingNode {
public:
  Neuron() { slot_by_input.fill(-1); }
  Neuron(const Neuron& other) : ArchivingNode(other), p(other.p), state(other.state),
    input_index(other.input_index), force_only(other.force_only),
    forced_steps(other.forced_steps), pre_indices(other.pre_indices), weights(other.weights),
    slot_by_input(other.slot_by_input) {}
  Parameters p;
  State state;
  long input_index{-1};
  bool force_only{};
  std::vector<long> forced_steps, pre_indices;
  std::vector<float> weights;
  std::array<long, input_count> slot_by_input;
  long pre_visits{}, post_visits{}, normalization_count{};
  using nest::Node::handle;
  using nest::Node::handles_test_event;

  size_t send_test_event(nest::Node& target, size_t receptor, nest::synindex, bool) override {
    nest::SpikeEvent event; event.set_sender(*this);
    return target.handles_test_event(event, receptor);
  }
  size_t handles_test_event(nest::SpikeEvent&, size_t receptor) override {
    if (receptor > input_count || (receptor != 0 && p.kind != 1))
      throw nest::UnknownReceptorType(receptor, get_name());
    return receptor;
  }
  void handle(nest::SpikeEvent& event) override {
    long lag = event.get_rel_delivery_steps(nest::kernel().simulation_manager.get_slice_origin());
    if (lag != 0 || event.get_delay_steps() != 1)
      throw nest::BadProperty("cpu_mnist requires one-resolution transport and one-tick slices");
    // Ordinary NEST SpikeEvents need not retain a sender node ID. The standard
    // static synapse's receptor encodes the input index, with port zero reserved
    // for recurrent current. This works across NEST worker threads.
    long source = static_cast<long>(event.get_rport()) - 1;
    if (source >= 0) {
      long slot = slot_by_input[source];
      if (p.kind != 1 || slot < 0 || event.get_multiplicity() != 1)
        throw nest::BadProperty("Unexpected MNIST feedforward spike or missing structural edge");
      arriving.push_back(slot);
    }
    else {
      float value = event.get_weight() * event.get_multiplicity();
      if (value >= 0) excitation.add_value(lag, value);
      else inhibition.add_value(lag, -value);
    }
  }
  void get_status(Dictionary& d) const override {
    ArchivingNode::get_status(d);
#define PUT_PARAM(n) d[#n] = double(p.n)
    PUT_PARAM(tau_m); PUT_PARAM(v_rest); PUT_PARAM(v_reset); PUT_PARAM(v_threshold);
    PUT_PARAM(e_exc); PUT_PARAM(e_inh); PUT_PARAM(theta_offset); PUT_PARAM(theta_plus);
    PUT_PARAM(ge_half_decay); PUT_PARAM(gi_half_decay); PUT_PARAM(ge_decay); PUT_PARAM(gi_decay); PUT_PARAM(theta_decay);
    PUT_PARAM(pre_tau); PUT_PARAM(post1_tau); PUT_PARAM(post2_tau); PUT_PARAM(potentiation);
    PUT_PARAM(depression); PUT_PARAM(weight_max); PUT_PARAM(pre_target); PUT_PARAM(pre_exponent);
    PUT_PARAM(post_exponent); PUT_PARAM(plasticity); PUT_PARAM(rate_hz);
#undef PUT_PARAM
    d["kind"] = p.kind; d["rule"] = p.rule; d["refractory_steps"] = p.refractory_steps;
#define PUT_STATE(n) d[#n] = double(state.n)
    PUT_STATE(V); PUT_STATE(ge); PUT_STATE(gi); PUT_STATE(theta); PUT_STATE(post_trace); PUT_STATE(last_input_current);
#undef PUT_STATE
    d["refractory"] = state.refractory; d["spike_count"] = state.spike_count;
    d["clock_step"] = state.clock_step; d["last_post"] = state.last_post;
    d["input_index"] = input_index;
    d["force_only"] = force_only; d["forced_steps"] = forced_steps;
    d["pre_indices"] = pre_indices;
    d["ff_weights"] = std::vector<double>(weights.begin(), weights.end());
    d["pre_visits"] = pre_visits; d["post_visits"] = post_visits;
    d["normalization_count"] = normalization_count;
    if (p.kind == 0 && input_index >= 0) {
      const auto& trace = inputs[input_index].trace[state.clock_step % 2];
      d["input_trace"] = double(trace.x); d["input_last_spike"] = trace.last_spike;
    }
  }
  void set_status(const Dictionary& d) override {
    auto next_p = p; auto next_s = state;
#define READ_PARAM(n) read_float(d, #n, next_p.n)
    READ_PARAM(tau_m); READ_PARAM(v_rest); READ_PARAM(v_reset); READ_PARAM(v_threshold);
    READ_PARAM(e_exc); READ_PARAM(e_inh); READ_PARAM(theta_offset); READ_PARAM(theta_plus);
    READ_PARAM(ge_half_decay); READ_PARAM(gi_half_decay); READ_PARAM(ge_decay); READ_PARAM(gi_decay); READ_PARAM(theta_decay);
    READ_PARAM(pre_tau); READ_PARAM(post1_tau); READ_PARAM(post2_tau); READ_PARAM(potentiation);
    READ_PARAM(depression); READ_PARAM(weight_max); READ_PARAM(pre_target); READ_PARAM(pre_exponent);
    READ_PARAM(post_exponent); READ_PARAM(plasticity); READ_PARAM(rate_hz);
#undef READ_PARAM
    d.update_value("kind", next_p.kind); d.update_value("rule", next_p.rule);
    d.update_value("refractory_steps", next_p.refractory_steps);
#define READ_STATE(n) read_float(d, #n, next_s.n)
    READ_STATE(V); READ_STATE(ge); READ_STATE(gi); READ_STATE(theta); READ_STATE(post_trace);
#undef READ_STATE
    d.update_value("refractory", next_s.refractory);
    if (next_p.kind < 0 || next_p.kind > 2 || next_p.rule < 1 || next_p.rule > 3
        || next_p.tau_m <= 0 || next_p.pre_tau <= 0 || next_p.post1_tau <= 0 || next_p.post2_tau <= 0
        || next_p.weight_max <= 0 || next_p.rate_hz < 0 || next_p.refractory_steps < 0)
      throw nest::BadProperty("Invalid cpu_mnist parameters");
    ArchivingNode::set_status(d);
    p = next_p; state = next_s;
    if (d.update_value("input_index", input_index)) {
      if (input_index < 0 || input_index >= long(input_count)) throw nest::BadProperty("Invalid MNIST input index");
      inputs[input_index] = SharedInput{};
    }
    d.update_value("force_only", force_only); d.update_value("forced_steps", forced_steps);
    if (!std::is_sorted(forced_steps.begin(), forced_steps.end())) throw nest::BadProperty("Forced steps must be sorted");
    if (d.update_value("pre_indices", pre_indices)) {
      slot_by_input.fill(-1);
      for (size_t i = 0; i < pre_indices.size(); ++i) {
        long pre = pre_indices[i];
        if (pre < 0 || pre >= long(input_count) || slot_by_input[pre] != -1)
          throw nest::BadProperty("Invalid/duplicate structural input index");
        slot_by_input[pre] = i;
      }
      if (!std::is_sorted(pre_indices.begin(), pre_indices.end())) throw nest::BadProperty("Input indices must be sorted");
    }
    if (d.known("ff_weights")) {
      std::vector<double> values; d.update_value("ff_weights", values);
      if (values.size() != pre_indices.size()) throw nest::BadProperty("MNIST weight/index shape mismatch");
      for (double value : values) if (!std::isfinite(value) || value < 0) throw nest::BadProperty("Invalid MNIST weight");
      weights.assign(values.begin(), values.end());
    }
    bool normalize = false;
    if (d.update_value("normalize_now", normalize) && normalize) {
      double sum = 0;
      for (float w : weights) sum += double(w);
      if (!std::isfinite(sum) || sum <= 0) throw nest::BadProperty("Invalid normalization column sum");
      double scale = 78.0 / sum;
      for (float& w : weights) w = float(double(w) * scale);
      ++normalization_count;
    }
  }
private:
  nest::RingBuffer excitation, inhibition;
  std::vector<long> arriving;
  double pre_decay{}, post_decay{};
  void init_buffers_() override { excitation.clear(); inhibition.clear(); arriving.clear(); }
  void pre_run_hook() override {
    if (nest::Time::get_resolution().get_ms() != 0.5)
      throw nest::BadProperty("cpu_mnist requires dt = 0.5 ms");
    if (p.kind == 0 && input_index < 0) throw nest::BadProperty("MNIST input index must be configured");
    if (weights.size() != pre_indices.size()) throw nest::BadProperty("MNIST weights not configured");
    pre_decay = std::exp(double(-0.5f / p.pre_tau));
    post_decay = std::exp(double(-0.5f / p.post1_tau));
  }
  void update(nest::Time const& origin, long from, long to) override {
    if (to - from != 1) throw nest::BadProperty("cpu_mnist requires a barrier after every timestep");
    auto rng = nest::get_vp_specific_rng(get_thread());
    for (long lag = from; lag < to; ++lag) {
      long step = origin.get_steps() + lag + 1;
      bool forced = std::binary_search(forced_steps.begin(), forced_steps.end(), step);
      bool fired = false;
      if (p.kind == 0) {
        // One Bernoulli draw per input per tick, broadcast to its whole fan-out.
        fired = force_only ? forced : (rng->drand() < p.rate_hz * 0.5f * 0.001f || forced);
        auto next = inputs[input_index].trace[(step - 1) % 2];
        if (p.rule != 3) {
          next.x = float(next.x * pre_decay);
          if (fired) next.x += 1.0f;
        }
        if (fired) { next.last_spike = step; ++state.spike_count; }
        inputs[input_index].trace[step % 2] = next;
      }
      else {
        float ex = static_cast<float>(excitation.get_value(lag));
        float in = static_cast<float>(inhibition.get_value(lag));
        float feedforward = 0;
        if (p.kind == 1) {
          long source_step = step - 1;
          long old_post = state.last_post < source_step ? state.last_post : state.previous_post;
          float old_y = std::fmax(0.0f, state.post_trace - (state.previous_fired ? 1.0f : 0.0f));
          std::sort(arriving.begin(), arriving.end());
          for (long slot : arriving) {
            feedforward += weights[slot];
            weights[slot] = on_pre(weights[slot], old_y, old_post, source_step, p);
            ++pre_visits;
          }
          arriving.clear();
          if (state.previous_fired) {
            for (size_t i = 0; i < weights.size(); ++i) {
              const auto& pre = inputs[pre_indices[i]].trace[source_step % 2];
              weights[i] = on_post(weights[i], pre.x, pre.last_spike,
                state.previous_post, source_step, p);
              ++post_visits;
            }
          }
        }
        state.last_input_current = feedforward;
        bool threshold = state.integrate(p, ex + feedforward, in);
        fired = force_only ? forced : (forced || threshold);
        if (p.kind == 1 && p.rule == 2) state.post_trace = float(state.post_trace * post_decay);
        if (fired) {
          state.fire(p, step);
          if (p.kind == 1 && p.rule == 2) state.post_trace += 1.0f;
        }
        state.previous_fired = fired;
      }
      state.clock_step = step;
      if (fired) {
        nest::Time time{nest::Time::step(step)};
        set_spiketime(time);
        nest::SpikeEvent event;
        nest::kernel().event_delivery_manager.send(*this, event, lag);
      }
    }
  }
};
class Module : public nest::NESTExtensionInterface {
public:
  void initialize() override { nest::register_node_model<Neuron>("cpu_mnist_neuron"); }
};
}
cpu_mnist::Module cpumnistmodule_LTX_module;
