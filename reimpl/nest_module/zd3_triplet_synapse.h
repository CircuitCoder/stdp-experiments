#ifndef ZD3_TRIPLET_SYNAPSE_H
#define ZD3_TRIPLET_SYNAPSE_H

#include <algorithm>
#include <cmath>
#include <deque>

#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"
#include "kernel_manager.h"
#include "nest_names.h"
#include "zd3_midpoint_neuron.h"

namespace zd3
{
void register_triplet_synapse( const std::string& name );

template < typename TargetIdentifierT >
class TripletSynapse : public nest::Connection< TargetIdentifierT >
{
public:
  using Base = nest::Connection< TargetIdentifierT >;
  using CommonPropertiesType = nest::CommonSynapseProperties;
  static constexpr nest::ConnectionModelProperties properties =
    nest::ConnectionModelProperties::HAS_DELAY | nest::ConnectionModelProperties::IS_PRIMARY;

  using Base::get_delay;
  using Base::get_delay_steps;
  using Base::get_rport;
  using Base::get_target;

  void get_status( Dictionary& d ) const;
  void set_status( const Dictionary& d, nest::ConnectorModel& cm );
  bool send( nest::Event& event, size_t thread, const nest::CommonSynapseProperties& );
  void set_weight( double weight ) { weight_ = weight; }

  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    using nest::ConnTestDummyNodeBase::handles_test_event;
    size_t handles_test_event( nest::SpikeEvent&, size_t ) override { return nest::invalid_port; }
    size_t handles_test_event( nest::DSSpikeEvent&, size_t ) override { return nest::invalid_port; }
  };

  void check_connection( nest::Node& source, nest::Node& target, size_t receptor,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy;
    Base::check_connection_( dummy, source, target, receptor );
    target.register_stdp_connection( last_pre_ - get_delay(), get_delay() );
  }

private:
  double weight_{ 1.0 };
  double pre_tau_{ 20.0 };
  double post1_tau_{ 20.0 };
  double post2_tau_{ 40.0 };
  double depression_{ 0.0001 };
  double potentiation_{ 0.01 };
  double max_weight_{ 1.0 };
  double last_pre_{ 0.0 };
  double last_post_{ -1.0 };
  bool has_pre_{ false };
};

template < typename T >
constexpr nest::ConnectionModelProperties TripletSynapse< T >::properties;

template < typename T >
bool TripletSynapse< T >::send( nest::Event& event, size_t thread,
  const nest::CommonSynapseProperties& )
{
  const double pre_time = event.get_stamp().get_ms();
  nest::Node* target = get_target( thread );
  auto* midpoint_target = dynamic_cast< MidpointNeuron* >( target );
  if ( not midpoint_target )
    throw nest::IllegalConnection( "zd3_triplet_synapse requires a zd3_midpoint_neuron target" );
  const double scale = midpoint_target->get_ff_scale();
  double effective_weight = weight_ * scale;
  std::deque< nest::histentry >::iterator begin;
  std::deque< nest::histentry >::iterator end;
  target->get_history( last_pre_, pre_time, &begin, &end );

  for ( auto it = begin; it != end; ++it )
  {
    const double post_time = it->t_;
    if ( has_pre_ )
    {
      const double pre_trace = std::exp( -( post_time - last_pre_ ) / pre_tau_ );
      const double post2_before = last_post_ < 0.0
        ? 0.0 : std::exp( -( post_time - last_post_ ) / post2_tau_ );
      effective_weight = std::clamp(
        effective_weight + potentiation_ * pre_trace * post2_before, 0.0, max_weight_ );
    }
    last_post_ = post_time;
  }

  const double post1 = last_post_ < 0.0
    ? 0.0 : std::exp( -( pre_time - last_post_ ) / post1_tau_ );
  const double transmission_weight = effective_weight;
  effective_weight = std::clamp( effective_weight - depression_ * post1, 0.0, max_weight_ );
  const double new_raw_weight = effective_weight / scale;
  midpoint_target->adjust_ff_raw_sum( new_raw_weight - weight_ );
  weight_ = new_raw_weight;
  last_pre_ = pre_time;
  has_pre_ = true;

  event.set_receiver( *target );
  event.set_weight( transmission_weight );
  event.set_delay_steps( get_delay_steps() );
  event.set_rport( get_rport() );
  event();
  return true;
}

template < typename T >
void TripletSynapse< T >::get_status( Dictionary& d ) const
{
  Base::get_status( d );
  d[ nest::names::weight ] = weight_;
  d[ "pre_tau" ] = pre_tau_;
  d[ "post1_tau" ] = post1_tau_;
  d[ "post2_tau" ] = post2_tau_;
  d[ "depression" ] = depression_;
  d[ "potentiation" ] = potentiation_;
  d[ nest::names::Wmax ] = max_weight_;
}

template < typename T >
void TripletSynapse< T >::set_status( const Dictionary& d, nest::ConnectorModel& cm )
{
  Base::set_status( d, cm );
  d.update_value( nest::names::weight, weight_ );
  d.update_value( "pre_tau", pre_tau_ );
  d.update_value( "post1_tau", post1_tau_ );
  d.update_value( "post2_tau", post2_tau_ );
  d.update_value( "depression", depression_ );
  d.update_value( "potentiation", potentiation_ );
  d.update_value( nest::names::Wmax, max_weight_ );
  if ( pre_tau_ <= 0 || post1_tau_ <= 0 || post2_tau_ <= 0 || max_weight_ <= 0 )
    throw nest::BadProperty( "ZD3 synapse time constants and Wmax must be positive" );
}
}

#endif
