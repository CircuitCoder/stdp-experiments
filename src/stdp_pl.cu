/* Power-law STDP synapse-group parameter storage. */

#include "cuda_error.h"
#include "stdp_pl.h"
#include "syn_model.h"

using namespace stdp_pl_ns;

int
STDPPL::_Init()
{
  type_ = i_stdp_pl_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_pl_param_name;
  CUDAMALLOCCTRL( "&d_param_arr_", &d_param_arr_, n_param_ * sizeof( float ) );
  SetParam( "tau_plus", 20.0 );
  SetParam( "tau_minus", 30.0 );
  SetParam( "lambda", 0.1 );
  SetParam( "alpha", 0.0513 );
  SetParam( "mu", 0.4 );
  return 0;
}
