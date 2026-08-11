#ifndef ZD3_MIDPOINT_NEURON_H
#define ZD3_MIDPOINT_NEURON_H

#include <atomic>

#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

namespace zd3
{
void register_midpoint_neuron( const std::string& name );

class MidpointNeuron : public nest::ArchivingNode
{
public:
  MidpointNeuron();
  MidpointNeuron( const MidpointNeuron& );

  using nest::Node::handle;
  using nest::Node::handles_test_event;

  size_t send_test_event( nest::Node&, size_t, nest::synindex, bool ) override;
  void handle( nest::SpikeEvent& ) override;
  void handle( nest::DataLoggingRequest& ) override;
  size_t handles_test_event( nest::SpikeEvent&, size_t ) override;
  size_t handles_test_event( nest::DataLoggingRequest&, size_t ) override;
  void get_status( Dictionary& ) const override;
  void set_status( const Dictionary& ) override;
  double get_ff_scale() const { return S_.ff_scale; }
  void adjust_ff_raw_sum( double delta )
  {
    std::atomic_ref< double >( S_.ff_raw_sum ).fetch_add( delta, std::memory_order_relaxed );
  }

private:
  void init_buffers_() override;
  void pre_run_hook() override;
  void update( nest::Time const&, long, long ) override;

  friend class nest::RecordablesMap< MidpointNeuron >;
  friend class nest::UniversalDataLogger< MidpointNeuron >;

  struct Parameters
  {
    double tau_m{ 100.0 };
    double v_rest{ -65.0 };
    double v_reset{ -65.0 };
    double v_threshold{ -52.0 };
    double e_exc{ 0.0 };
    double e_inh{ -100.0 };
    double tau_ge{ 1.0 };
    double tau_gi{ 2.0 };
    double t_ref{ 5.0 };
    double theta_offset{ 20.0 };
    double theta_plus{ 0.05 };
    double theta_tau{ 1.0e7 };
    double plasticity{ 1.0 };

    void get( Dictionary& ) const;
    void set( const Dictionary&, nest::Node* );
  };

  struct State
  {
    double v{ -105.0 };
    double ge{ 0.0 };
    double gi{ 0.0 };
    double theta{ 20.0 };
    long refractory_steps{ 0 };
    long spike_count{ 0 };
    double ff_scale{ 1.0 };
    double ff_raw_sum{ 0.0 };

    void get( Dictionary& ) const;
    void set( const Dictionary&, nest::Node* );
  };

  struct Buffers
  {
    explicit Buffers( MidpointNeuron& n ) : logger( n ) {}
    Buffers( const Buffers&, MidpointNeuron& n ) : logger( n ) {}
    nest::RingBuffer excitation;
    nest::RingBuffer inhibition;
    nest::UniversalDataLogger< MidpointNeuron > logger;
  };

  struct Variables
  {
    double ge_half_decay{};
    double gi_half_decay{};
    double ge_decay{};
    double gi_decay{};
    double theta_decay{};
    double dt{};
    long refractory_steps{};
  };

  double get_v_() const { return S_.v; }
  double get_ge_() const { return S_.ge; }
  double get_gi_() const { return S_.gi; }
  double get_theta_() const { return S_.theta; }

  Parameters P_;
  State S_;
  Variables V_;
  Buffers B_;
  static nest::RecordablesMap< MidpointNeuron > recordablesMap_;
};

inline size_t
MidpointNeuron::send_test_event( nest::Node& target, size_t receptor, nest::synindex, bool )
{
  nest::SpikeEvent event;
  event.set_sender( *this );
  return target.handles_test_event( event, receptor );
}

inline size_t
MidpointNeuron::handles_test_event( nest::SpikeEvent&, size_t receptor )
{
  if ( receptor != 0 )
    throw nest::UnknownReceptorType( receptor, get_name() );
  return 0;
}

inline size_t
MidpointNeuron::handles_test_event( nest::DataLoggingRequest& event, size_t receptor )
{
  if ( receptor != 0 )
    throw nest::UnknownReceptorType( receptor, get_name() );
  return B_.logger.connect_logging_device( event, recordablesMap_ );
}
}

#endif
