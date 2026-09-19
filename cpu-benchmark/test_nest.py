"""Deterministic probes against independently calculated steps and STDP."""
import os
from math import exp
from pathlib import Path
import sys
import numpy as np
import pytest

import importlib.util
spec=importlib.util.spec_from_file_location('cpu_nest_runner',Path(__file__).with_name('run_nest.py'))
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
parameters=runner.parameters


@pytest.fixture
def nest_api():
    prefix=os.environ.get('CPU_BENCH_NEST_PREFIX')
    module=os.environ.get('CPU_BENCH_NEST_MODULE')
    if not prefix or not module:pytest.skip('Set source-built NEST prefix and custom module')
    for p in Path(prefix).glob('lib*/python*/site-packages'):sys.path.insert(0,str(p))
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':0.1,'local_num_threads':2,'rng_seed':20260724})
    nest.Install(str(Path(module).resolve().with_suffix('')))
    return nest


def test_voltage_step_and_refractory(nest_api):
    nest=nest_api
    p=parameters('additive')|{'external_mean':0.0,'threshold':100000.0}
    state={'V':3.0,'Iex':2.0,'dIex':4.0,'Iin':-1.0,'dIin':-3.0}
    neurons=nest.Create('cpu_brunel_neuron',2,p|state)
    neurons[1].set({'refractory':2})
    nest.Simulate(0.1)
    expected=p['p31']*(4-3)+p['p32']*(2-1)+p['p33']*3
    assert neurons[0].get('V')==pytest.approx(expected,abs=2e-6)
    assert neurons[1].get('V')==3.0
    for n in neurons:
        assert n.get('dIex')==pytest.approx(4*exp(-0.1/0.32582722403722841),abs=2e-6)
    assert neurons[1].get('refractory')==1


def test_inhibition_arrives_on_following_step(nest_api):
    nest=nest_api
    p=parameters('additive')|{'external_mean':0.0,'threshold':100000.0}
    pre=nest.Create('cpu_brunel_neuron',1,p|{'forced_steps':[2]})
    post=nest.Create('cpu_brunel_neuron',1,p)
    nest.Connect(pre,post,syn_spec={'synapse_model':'static_synapse_hpc','weight':-4.0,'delay':0.1})
    nest.Simulate(0.2)
    assert post.get('dIin')==0.0
    nest.Simulate(0.1)
    assert post.get('dIin')==-4.0
    assert post.get('V')==0.0
    nest.Simulate(0.1)
    assert post.get('V')==pytest.approx(-4*p['p31'],abs=2e-6)


@pytest.mark.parametrize('rule',['additive','morrison'])
@pytest.mark.parametrize('synapse_model',['cpu_brunel_stdp','cpu_brunel_stdp_hpc'])
def test_arrival_weight_and_simultaneous_tie(nest_api,rule,synapse_model):
    nest=nest_api
    p=parameters(rule)|{'external_mean':0.0,'threshold':100000.0,
        'p11':0.0,'p21':0.0,'p22':0.0,'p31':0.0,'p32':0.0,'p33':1.0,
        'epsc_initial':1.0,'refractory_steps':0}
    pre_steps={2,6,8,9,12};post_steps={4,6,9,10,12}
    pre=nest.Create('cpu_brunel_neuron',1,p|{'forced_steps':sorted(pre_steps)})
    post=nest.Create('cpu_brunel_neuron',1,p|{'forced_steps':sorted(post_steps)})
    nest.Connect(pre,post,syn_spec={'synapse_model':synapse_model,'weight':45.0,'delay':0.1})
    weight,x,y=45.0,0.0,0.0
    pending=0.0
    for step in range(1,16):
        nest.Simulate(0.1)
        assert post.get('dIex')==pytest.approx(pending,rel=3e-6,abs=3e-6)
        x*=exp(-0.1/p['tau_plus']);y*=exp(-0.1/p['tau_minus'])
        old_y=y
        if step in post_steps:
            if rule=='additive':weight=np.clip(weight+p['learning_rate']*p['weight_max']*x,0,p['weight_max'])
            else:weight+=p['learning_rate']*weight**p['mu']*x
            y+=1
        pending=0.0
        if step in pre_steps:
            if rule=='additive':weight=np.clip(weight-p['depression_ratio']*p['learning_rate']*p['weight_max']*old_y,0,p['weight_max'])
            else:weight=max(0,weight-p['learning_rate']*p['depression_ratio']*weight*old_y)
            x+=1
            pending=weight*np.sqrt(20)
