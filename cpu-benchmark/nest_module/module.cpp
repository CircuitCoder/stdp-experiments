#include "mechanics.h"
#include <cstdint>
#include <vector>
#include "archiving_node.h"
#include "connection.h"
#include "connector_model.h"
#include "dict_util.h"
#include "event.h"
#include "kernel_manager.h"
#include "model_manager_impl.h"
#include "nest_extension_interface.h"
#include "random_generators.h"
#include "ring_buffer.h"

namespace cpu_brunel {
inline void read_float(const Dictionary& d, const std::string& name, float& value) {
  double temporary = value;
  d.update_value(name,temporary);
  value = static_cast<float>(temporary);
}

class Neuron : public nest::ArchivingNode {
public:
  Neuron() = default;
  Neuron(const Neuron& other) : ArchivingNode(other), p(other.p), state(other.state),
    learning(other.learning), forced_steps(other.forced_steps) {}
  NeuronParameters p;
  NeuronState state;
  LearningParameters learning;
  // Kept for this bounded benchmark so lazy synapses and diagnostic reads use
  // the same post events. Each history is owned by its target's NEST thread.
  std::vector<double> spike_history;
  std::vector<long> forced_steps;
  using nest::Node::handle;
  using nest::Node::handles_test_event;
  size_t send_test_event(nest::Node& target, size_t receptor, nest::synindex, bool) override {
    nest::SpikeEvent event; event.set_sender(*this);
    return target.handles_test_event(event,receptor);
  }
  size_t handles_test_event(nest::SpikeEvent&, size_t receptor) override {
    if (receptor != 0) throw nest::UnknownReceptorType(receptor,get_name());
    return 0;
  }
  void handle(nest::SpikeEvent& event) override {
    const long lag = event.get_rel_delivery_steps(nest::kernel().simulation_manager.get_slice_origin());
    const double value = event.get_weight()*event.get_multiplicity();
    if (value >= 0) excitation.add_value(lag,value);
    else inhibition.add_value(lag,value);
  }
  void get_status(Dictionary& d) const override {
    ArchivingNode::get_status(d);
#define PUT_PARAM(n) d[#n] = double(p.n)
    PUT_PARAM(p11); PUT_PARAM(p21); PUT_PARAM(p22); PUT_PARAM(p31); PUT_PARAM(p32); PUT_PARAM(p33);
    PUT_PARAM(epsc_initial); PUT_PARAM(external_weight); PUT_PARAM(external_mean); PUT_PARAM(threshold); PUT_PARAM(reset);
#undef PUT_PARAM
    d["refractory_steps"] = p.refractory_steps;
#define PUT_STATE(n) d[#n] = double(state.n)
    PUT_STATE(V); PUT_STATE(Iex); PUT_STATE(dIex); PUT_STATE(Iin); PUT_STATE(dIin);
#undef PUT_STATE
    d["refractory"] = state.refractory; d["spike_count"] = state.spike_count;
    d["forced_steps"] = forced_steps;
#define PUT_LEARN(n) d[#n] = double(learning.n)
    PUT_LEARN(learning_rate); PUT_LEARN(depression_ratio); PUT_LEARN(mu); PUT_LEARN(weight_max);
    PUT_LEARN(tau_plus); PUT_LEARN(tau_minus);
#undef PUT_LEARN
    d["additive"] = learning.additive;
  }
  void set_status(const Dictionary& d) override {
    auto new_p = p; auto new_state = state; auto new_learning = learning;
#define READ_PARAM(n) read_float(d,#n,new_p.n)
    READ_PARAM(p11); READ_PARAM(p21); READ_PARAM(p22); READ_PARAM(p31); READ_PARAM(p32); READ_PARAM(p33);
    READ_PARAM(epsc_initial); READ_PARAM(external_weight); READ_PARAM(external_mean); READ_PARAM(threshold); READ_PARAM(reset);
#undef READ_PARAM
    d.update_value("refractory_steps",new_p.refractory_steps);
#define READ_STATE(n) read_float(d,#n,new_state.n)
    READ_STATE(V); READ_STATE(Iex); READ_STATE(dIex); READ_STATE(Iin); READ_STATE(dIin);
#undef READ_STATE
    d.update_value("refractory",new_state.refractory); d.update_value("spike_count",new_state.spike_count);
#define READ_LEARN(n) read_float(d,#n,new_learning.n)
    READ_LEARN(learning_rate); READ_LEARN(depression_ratio); READ_LEARN(mu); READ_LEARN(weight_max);
    READ_LEARN(tau_plus); READ_LEARN(tau_minus);
#undef READ_LEARN
    d.update_value("additive",new_learning.additive);
    if (new_p.external_mean < 0 || new_p.external_mean > 50 || new_p.refractory_steps < 0
        || new_learning.tau_plus <= 0 || new_learning.tau_minus <= 0)
      throw nest::BadProperty("Invalid cpu_brunel parameters");
    ArchivingNode::set_status(d);
    d.update_value("forced_steps",forced_steps);
    p=new_p; state=new_state; learning=new_learning;
  }
private:
  nest::RingBuffer excitation, inhibition;
  float poisson_limit{};
  void init_buffers_() override { excitation.clear(); inhibition.clear(); spike_history.clear(); }
  void pre_run_hook() override { poisson_limit=std::exp(-p.external_mean); }
  void update(nest::Time const& origin,long from,long to) override {
    auto rng=nest::get_vp_specific_rng(get_thread());
    for (long lag=from;lag<to;++lag) {
      unsigned int external=0;
      if (p.external_mean>0) {
        float product=static_cast<float>(rng->drand());
        while (product>poisson_limit) { ++external; product*=static_cast<float>(rng->drand()); }
      }
      bool fired=state.step(p,static_cast<float>(excitation.get_value(lag)),
        static_cast<float>(inhibition.get_value(lag)),external);
      long step=origin.get_steps()+lag+1;
      fired = fired || std::binary_search(forced_steps.begin(),forced_steps.end(),step);
      if (fired) {
        state.spike(p);
        nest::Time time{nest::Time::step(step)};
        spike_history.push_back(time.get_ms());
        set_spiketime(time);
        nest::SpikeEvent event;
        nest::kernel().event_delivery_manager.send(*this,event,lag);
      }
    }
  }
};

template<typename TargetIdentifierT>
class Synapse : public nest::Connection<TargetIdentifierT> {
public:
  using Base=nest::Connection<TargetIdentifierT>;
  using CommonPropertiesType=nest::CommonSynapseProperties;
  static constexpr nest::ConnectionModelProperties properties =
    nest::ConnectionModelProperties::HAS_DELAY | nest::ConnectionModelProperties::IS_PRIMARY
    | nest::ConnectionModelProperties::SUPPORTS_HPC;
  class Dummy : public nest::ConnTestDummyNodeBase {
  public:
    using nest::ConnTestDummyNodeBase::handles_test_event;
    size_t handles_test_event(nest::SpikeEvent&,size_t) override { return nest::invalid_port; }
  };
  void check_connection(nest::Node& source,nest::Node& target,size_t receptor,const CommonPropertiesType&) {
    Dummy dummy;
    Base::check_connection_(dummy,source,target,receptor);
    target_=dynamic_cast<Neuron*>(&target);
    if (!target_) throw nest::IllegalConnection("cpu_brunel_stdp requires cpu_brunel_neuron target");
    if (Base::get_delay_steps()!=1) throw nest::BadProperty("cpu_brunel_stdp requires one-step arrival latency");
  }
  void set_weight(double value) { state_.weight=static_cast<float>(value); }
  void get_status(Dictionary& d) const {
    Base::get_status(d);
    auto snapshot=state_; auto cursor=cursor_;
    if (target_) posts(snapshot,cursor,nest::kernel().simulation_manager.get_time().get_ms());
    d[nest::names::weight]=double(snapshot.weight);
    d["pre_trace"]=double(snapshot.pre); d["post_trace"]=double(snapshot.post);
  }
  void set_status(const Dictionary& d,nest::ConnectorModel& cm) {
    Base::set_status(d,cm);
    read_float(d,"weight",state_.weight);
    if (state_.weight<0) throw nest::BadProperty("Plastic weight must be nonnegative");
  }
  bool send(nest::Event& event,size_t thread,const CommonPropertiesType&) {
    auto* target=static_cast<Neuron*>(Base::get_target(thread));
    target_=target;
    double t=event.get_stamp().get_ms();
    bool simultaneous=posts(state_,cursor_,t);
    float weight=state_.on_pre(static_cast<float>(t),simultaneous,target->learning);
    event.set_receiver(*target);
    // Connection weight is already converted into a derivative increment by
    // the runner's scale, preserving raw plastic weight units in state_.
    event.set_weight(double(target->p.epsc_initial * delivery_scale_ * weight));
    event.set_delay_steps(Base::get_delay_steps()); event.set_rport(Base::get_rport()); event();
    return true;
  }
  // Fixed by the validated baseline; raw weights remain in pA.
  static constexpr float delivery_scale_=4.47213595499958f;
private:
  LearningState state_;
  std::uint32_t cursor_{};
  Neuron* target_{};
  bool posts(LearningState& state,std::uint32_t& cursor,double until) const {
    bool simultaneous=false;
    const auto& history=target_->spike_history;
    while (cursor<history.size() && history[cursor]<=until+1e-8) {
      double t=history[cursor++];
      state.on_post(static_cast<float>(t),target_->learning);
      simultaneous=std::abs(t-until)<1e-8;
    }
    return simultaneous;
  }
};

class Module : public nest::NESTExtensionInterface {
public:
  void initialize() override {
    nest::register_node_model<Neuron>("cpu_brunel_neuron");
    nest::register_connection_model<Synapse>("cpu_brunel_stdp");
  }
};
}
cpu_brunel::Module cpubrunelmodule_LTX_module;
