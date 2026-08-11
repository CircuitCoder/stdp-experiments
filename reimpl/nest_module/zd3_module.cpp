#include "nest_extension_interface.h"

#include "zd3_midpoint_neuron.h"
#include "zd3_triplet_synapse.h"

namespace zd3
{
class ZD3Module : public nest::NESTExtensionInterface
{
public:
  void initialize() override
  {
    register_midpoint_neuron( "zd3_midpoint_neuron" );
    register_triplet_synapse( "zd3_triplet_synapse" );
  }
};
}

zd3::ZD3Module zd3module_LTX_module;

