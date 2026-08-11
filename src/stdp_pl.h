/*
 * Power-law STDP update used by the Morrison Brunel benchmark port.
 *
 * NEST GPU's synapse callback supplies the nearest signed spike interval and
 * does not store per-connection traces. This therefore implements the NEST
 * stdp_pl weight dependence with nearest-pair, rather than all-to-all, timing.
 */

#ifndef STDP_PL_H
#define STDP_PL_H

#include <cmath>
#include <string>

namespace stdp_pl_ns
{
enum ParamIndexes
{
  i_tau_plus = 0,
  i_tau_minus,
  i_lambda,
  i_alpha,
  i_mu,
  N_PARAM
};

const std::string stdp_pl_param_name[ N_PARAM ] = {
  "tau_plus", "tau_minus", "lambda", "alpha", "mu"
};

__device__ __forceinline__ void
STDPPLUpdate( float* weight_pt, float Dt, float* param )
{
  const double w = *weight_pt;
  double updated;
  if ( Dt >= 0.0f )
  {
    updated = w + param[ i_lambda ] * pow( w, param[ i_mu ] )
      * exp( -( double ) Dt / param[ i_tau_plus ] );
  }
  else
  {
    updated = w - param[ i_lambda ] * param[ i_alpha ] * w
      * exp( ( double ) Dt / param[ i_tau_minus ] );
  }
  *weight_pt = ( float ) ( updated > 0.0 ? updated : 0.0 );
}
} // namespace stdp_pl_ns

#endif
