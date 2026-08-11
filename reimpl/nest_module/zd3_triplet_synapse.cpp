#include "zd3_triplet_synapse.h"

#include "model_manager_impl.h"
#include "nest_impl.h"

void
zd3::register_triplet_synapse( const std::string& name )
{
  nest::register_connection_model< TripletSynapse >( name );
}
