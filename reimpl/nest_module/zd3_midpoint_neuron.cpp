#include "zd3_midpoint_neuron.h"

#include <algorithm>
#include <cmath>

#include "dict_util.h"
#include "nest_impl.h"
#include "universal_data_logger_impl.h"

using namespace nest;

namespace
{
const std::string tau_m_name( "tau_m" );
const std::string v_rest_name( "v_rest" );
const std::string e_exc_name( "e_exc" );
const std::string e_inh_name( "e_inh" );
const std::string tau_ge_name( "tau_ge" );
const std::string tau_gi_name( "tau_gi" );
const std::string theta_offset_name( "theta_offset" );
const std::string theta_plus_name( "theta_plus" );
const std::string theta_tau_name( "theta_tau" );
const std::string plasticity_name( "plasticity" );
const std::string ge_name( "ge" );
const std::string gi_name( "gi" );
const std::string theta_name( "theta" );
const std::string spike_count_name( "spike_count" );
const std::string ff_scale_name( "ff_scale" );
const std::string ff_raw_sum_name( "ff_raw_sum" );
}

nest::RecordablesMap< zd3::MidpointNeuron > zd3::MidpointNeuron::recordablesMap_;

void
zd3::register_midpoint_neuron( const std::string& name )
{
  nest::register_node_model< MidpointNeuron >( name );
}

namespace nest
{
template <>
void RecordablesMap< zd3::MidpointNeuron >::create()
{
  insert_( names::V_m, &zd3::MidpointNeuron::get_v_ );
  insert_( ge_name, &zd3::MidpointNeuron::get_ge_ );
  insert_( gi_name, &zd3::MidpointNeuron::get_gi_ );
  insert_( theta_name, &zd3::MidpointNeuron::get_theta_ );
}
}

zd3::MidpointNeuron::MidpointNeuron()
  : ArchivingNode(), P_(), S_(), V_(), B_( *this )
{
  recordablesMap_.create();
}

zd3::MidpointNeuron::MidpointNeuron( const MidpointNeuron& other )
  : ArchivingNode( other ), P_( other.P_ ), S_( other.S_ ), V_( other.V_ ), B_( other.B_, *this )
{}

void zd3::MidpointNeuron::Parameters::get( Dictionary& d ) const
{
  d[ tau_m_name ] = tau_m;
  d[ v_rest_name ] = v_rest;
  d[ names::V_reset ] = v_reset;
  d[ names::V_th ] = v_threshold;
  d[ e_exc_name ] = e_exc;
  d[ e_inh_name ] = e_inh;
  d[ tau_ge_name ] = tau_ge;
  d[ tau_gi_name ] = tau_gi;
  d[ names::t_ref ] = t_ref;
  d[ theta_offset_name ] = theta_offset;
  d[ theta_plus_name ] = theta_plus;
  d[ theta_tau_name ] = theta_tau;
  d[ plasticity_name ] = plasticity;
}

void zd3::MidpointNeuron::Parameters::set( const Dictionary& d, Node* node )
{
  update_value_param( d, tau_m_name, tau_m, node );
  update_value_param( d, v_rest_name, v_rest, node );
  update_value_param( d, names::V_reset, v_reset, node );
  update_value_param( d, names::V_th, v_threshold, node );
  update_value_param( d, e_exc_name, e_exc, node );
  update_value_param( d, e_inh_name, e_inh, node );
  update_value_param( d, tau_ge_name, tau_ge, node );
  update_value_param( d, tau_gi_name, tau_gi, node );
  update_value_param( d, names::t_ref, t_ref, node );
  update_value_param( d, theta_offset_name, theta_offset, node );
  update_value_param( d, theta_plus_name, theta_plus, node );
  update_value_param( d, theta_tau_name, theta_tau, node );
  update_value_param( d, plasticity_name, plasticity, node );
  if ( tau_m <= 0 || tau_ge <= 0 || tau_gi <= 0 || theta_tau <= 0 || t_ref < 0 )
    throw BadProperty( "ZD3 time constants must be positive and t_ref non-negative" );
}

void zd3::MidpointNeuron::State::get( Dictionary& d ) const
{
  d[ names::V_m ] = v;
  d[ ge_name ] = ge;
  d[ gi_name ] = gi;
  d[ theta_name ] = theta;
  d[ spike_count_name ] = spike_count;
  d[ ff_scale_name ] = ff_scale;
  d[ ff_raw_sum_name ] = ff_raw_sum;
}

void zd3::MidpointNeuron::State::set( const Dictionary& d, Node* node )
{
  update_value_param( d, names::V_m, v, node );
  update_value_param( d, ge_name, ge, node );
  update_value_param( d, gi_name, gi, node );
  update_value_param( d, theta_name, theta, node );
  update_value_param( d, spike_count_name, spike_count, node );
  update_value_param( d, ff_scale_name, ff_scale, node );
  update_value_param( d, ff_raw_sum_name, ff_raw_sum, node );
  if ( ff_scale <= 0.0 || ff_raw_sum < 0.0 )
    throw BadProperty( "ZD3 feedforward scale must be positive and raw sum non-negative" );
}

void zd3::MidpointNeuron::get_status( Dictionary& d ) const
{
  P_.get( d );
  S_.get( d );
  ArchivingNode::get_status( d );
  d[ names::recordables ] = recordablesMap_.get_list();
}

void zd3::MidpointNeuron::set_status( const Dictionary& d )
{
  Parameters p = P_;
  State s = S_;
  p.set( d, this );
  s.set( d, this );
  ArchivingNode::set_status( d );
  P_ = p;
  S_ = s;
}

void zd3::MidpointNeuron::init_buffers_()
{
  B_.excitation.clear();
  B_.inhibition.clear();
  B_.logger.reset();
}

void zd3::MidpointNeuron::pre_run_hook()
{
  B_.logger.init();
  V_.dt = Time::get_resolution().get_ms();
  V_.ge_half_decay = std::exp( -0.5 * V_.dt / P_.tau_ge );
  V_.gi_half_decay = std::exp( -0.5 * V_.dt / P_.tau_gi );
  V_.ge_decay = std::exp( -V_.dt / P_.tau_ge );
  V_.gi_decay = std::exp( -V_.dt / P_.tau_gi );
  V_.theta_decay = std::exp( -V_.dt / P_.theta_tau );
  V_.refractory_steps = Time( Time::ms( P_.t_ref ) ).get_steps();
}

void zd3::MidpointNeuron::update( Time const& origin, long from, long to )
{
  for ( long lag = from; lag < to; ++lag )
  {
    S_.ge += B_.excitation.get_value( lag );
    S_.gi += B_.inhibition.get_value( lag );

    if ( S_.refractory_steps > 0 )
      --S_.refractory_steps;
    else
    {
      const double ge_mid = S_.ge * V_.ge_half_decay;
      const double gi_mid = S_.gi * V_.gi_half_decay;
      const double g_mid = 1.0 + ge_mid + gi_mid;
      const double v_inf = ( P_.v_rest + ge_mid * P_.e_exc + gi_mid * P_.e_inh ) / g_mid;
      S_.v = v_inf + ( S_.v - v_inf ) * std::exp( -V_.dt * g_mid / P_.tau_m );
      S_.ge *= V_.ge_decay;
      S_.gi *= V_.gi_decay;
      if ( P_.plasticity != 0.0 )
        S_.theta *= V_.theta_decay;
    }

    if ( S_.refractory_steps == 0 && S_.v > S_.theta - P_.theta_offset + P_.v_threshold )
    {
      S_.v = P_.v_reset;
      S_.refractory_steps = V_.refractory_steps;
      if ( P_.plasticity != 0.0 )
        S_.theta += P_.theta_plus;
      ++S_.spike_count;
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
      SpikeEvent event;
      kernel().event_delivery_manager.send( *this, event, lag );
    }

    B_.logger.record_data( origin.get_steps() + lag );
  }
}

void zd3::MidpointNeuron::handle( SpikeEvent& event )
{
  const double value = event.get_weight() * event.get_multiplicity();
  const long lag = event.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() );
  if ( value >= 0.0 )
    B_.excitation.add_value( lag, value );
  else
    B_.inhibition.add_value( lag, -value );
}

void zd3::MidpointNeuron::handle( DataLoggingRequest& event )
{
  B_.logger.handle( event );
}
