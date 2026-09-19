#!/usr/bin/env python3
"""Multithreaded NEST port of the current GeNN Brunel workload."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'brunel'))
sys.path.insert(0,str(ROOT/'reimpl'))
from ports.common import make_genn_default_model, alpha_propagator, JE_PA, DT_MS
from zd3.io import sha256_file
from energy import PackageEnergy


def parameters(rule):
    spec = make_genn_default_model(rule)
    p=alpha_propagator()
    return {k:p[k] for k in ['p11','p21','p22','p31','p32','p33','epsc_initial']} | {
        'external_weight':JE_PA, 'external_mean':spec.external_rate_hz*DT_MS/1000,
        'threshold':20.0,'reset':0.0,'refractory_steps':5,
        'additive':rule=='additive','learning_rate':spec.rule.learning_rate,
        'depression_ratio':spec.rule.depression_ratio,'mu':spec.rule.mu_plus,
        'weight_max':spec.rule.weight_max_pa or 1e30,'tau_plus':20.0,'tau_minus':30.0}


def counts(population):
    return np.asarray(population.get('spike_count'),dtype=np.int64)


def execute(nest,args,repetition,diagnostic):
    spec=make_genn_default_model(args.rule)
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':DT_MS,'local_num_threads':args.threads,
        'rng_seed':args.seed,'print_time':False})
    nest.Install(str(args.module.with_suffix('')))
    nest.set_verbosity('M_WARNING')
    begin=time.perf_counter()
    exc=nest.Create('cpu_brunel_neuron',spec.ne,parameters(args.rule))
    inh=nest.Create('cpu_brunel_neuron',spec.ni,parameters(args.rule))
    rng=np.random.default_rng(args.seed)
    exc.set({'V':rng.normal(5.7,7.2,spec.ne).astype(np.float32)})
    inh.set({'V':rng.normal(5.7,7.2,spec.ni).astype(np.float32)})
    scale=np.float32(alpha_propagator()['epsc_initial'])*np.float32(spec.recurrent_delivery_scale)
    for source,target,degree,plastic,inhibitory in [
        (exc,exc,spec.ce,True,False),(exc,inh,spec.ce,False,False),
        (inh,exc,spec.ci,False,True),(inh,inh,spec.ci,False,True)]:
        weight=JE_PA if plastic else float(scale*np.float32(JE_PA)*np.float32(-spec.rule.inhibitory_weight_ratio if inhibitory else 1.0))
        nest.Connect(source,target,
            {'rule':'fixed_indegree','indegree':degree,'allow_autapses':False,'allow_multapses':True},
            {'synapse_model':'cpu_brunel_stdp_hpc' if plastic else 'static_synapse_hpc',
             'weight':weight,'delay':DT_MS})
    construction=time.perf_counter()-begin
    print(f'CONSTRUCTED rule={args.rule} threads={args.threads} seconds={construction:.3f}',flush=True)
    nest.Prepare()
    try:
        previous=counts(exc)
        for _ in range(5):
            nest.Run(20.0)
            current=counts(exc)
            if float((current-previous).mean())/0.02>100:
                raise RuntimeError('Presimulation exceeded 100 Hz E population rate')
            previous=current
        print(f'PRESIM_COMPLETE rule={args.rule} threads={args.threads}',flush=True)
        before=counts(exc),counts(inh)
        energy=PackageEnergy()
        load_before=os.getloadavg()
        energy.start()
        started=time.perf_counter()
        bins=[]
        if diagnostic:
            previous=before[0]
            for offset in range(0,10000,30):
                nest.Run(min(30,10000-offset)*DT_MS)
                current=counts(exc)
                if offset+30<=10000: bins.append(int((current-previous).sum()))
                previous=current
                if (offset+30)%300==0:
                    rate=sum(bins[-10:])/spec.ne/0.03
                    if rate>100:raise RuntimeError(f'Diagnostic exceeded 100 Hz: {rate}')
                if (offset+30)%3000==0:
                    print(f'DIAGNOSTIC rule={args.rule} elapsed_ms={(offset+30)*DT_MS:.1f}',flush=True)
        else: nest.Run(1000.0)
        after=counts(exc),counts(inh)
        wall=time.perf_counter()-started
        measured_energy=energy.stop()
        spikes=[after[i]-before[i] for i in (0,1)]
        result={'repetition':repetition,'diagnostic_mode':diagnostic,'threads':nest.local_num_threads,
            'construction_wall_seconds':construction,'wall_seconds':wall,'simulation_steps':10000,
            'us_per_step':wall*100,'energy':measured_energy,
            'microjoules_per_step':measured_energy['joules']*100 if measured_energy['joules'] is not None else None,
            'exc_spikes':int(spikes[0].sum()),'inh_spikes':int(spikes[1].sum()),
            'exc_rate_hz':float(spikes[0].mean()),'inh_rate_hz':float(spikes[1].mean()),
            'load_average_before':load_before,'load_average_after':os.getloadavg()}
        if not 0.1<result['exc_rate_hz']<100 or not 0.1<result['inh_rate_hz']<100:
            raise RuntimeError(f'Silence/runaway: {result}')
        if diagnostic:
            ranges={}
            for name,pop in [('E',exc),('I',inh)]:
                ranges[name]={}
                for field in ['V','Iex','Iin','dIex','dIin']:
                    values=np.asarray(pop.get(field))
                    if not np.all(np.isfinite(values)):raise RuntimeError(f'Nonfinite {name}/{field}')
                    ranges[name][field]=[float(values.min()),float(values.max())]
            connections=nest.GetConnections(exc,exc,synapse_model='cpu_brunel_stdp_hpc')
            weights=np.asarray(connections[::max(1,len(connections)//100000)].get('weight'))
            if not np.all(np.isfinite(weights)) or weights.min()<0:raise RuntimeError('Invalid weights')
            if spec.rule.weight_max_pa and weights.max()>spec.rule.weight_max_pa*1.00001:
                raise RuntimeError('Weight bound exceeded')
            result['diagnostic']={'ranges':ranges,'weight_min':float(weights.min()),
                'weight_max':float(weights.max()),'weight_mean':float(weights.mean()),'weight_std':float(weights.std()),
                'plastic_synapses':len(connections),'exc_spikes_per_neuron':spikes[0].tolist(),
                'population_fano_3ms':float(np.var(bins)/np.mean(bins)),'guards_passed':True}
        return result
    finally: nest.Cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--module',type=Path,required=True)
    p.add_argument('--nest-prefix',type=Path,required=True)
    p.add_argument('--threads',type=int,default=len(os.sched_getaffinity(0)))
    p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--rule',choices=('additive','morrison'),required=True)
    p.add_argument('--seed',type=int,default=20260724)
    args=p.parse_args()
    args.module=args.module.resolve()
    if min(args.threads,args.repetitions,args.seed)<=0:p.error('positive counts and seed required')
    candidates=list(args.nest_prefix.glob('lib*/python*/site-packages'))
    if len(candidates)!=1:raise RuntimeError(f'Cannot find unique PyNEST installation: {candidates}')
    sys.path.insert(0,str(candidates[0]))
    import nest
    args.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (args.output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    sources=[Path(__file__),Path(__file__).with_name('nest_module')/'module.cpp',
        Path(__file__).with_name('nest_module')/'mechanics.h',ROOT/'brunel/ports/common.py',
        Path(__file__).with_name('energy.py'),Path(__file__).with_name('shell.nix'),
        Path(__file__).with_name('build_nest.sh'),Path(__file__).with_name('nest_module')/'CMakeLists.txt']
    write('manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
        'cwd':str(Path.cwd()),'nest_version':nest.__version__,'host':platform.uname()._asdict(),
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_sha256':{str(f.relative_to(ROOT)):sha256_file(f) for f in sources},
        'module_sha256':sha256_file(args.module),'environment':{k:os.environ.get(k) for k in
        ['PATH','LD_LIBRARY_PATH','CXX','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NIX_ENFORCE_NO_NATIVE',
         'OMP_WAIT_POLICY','GOMP_SPINCOUNT','OMP_PROC_BIND','OMP_PLACES']},
        'model':make_genn_default_model(args.rule).as_dict(),'precision':'FP32 neuron and plastic state; NEST event accumulation is FP64',
        'protocol':'100 ms presimulation + 1000 ms training; native NEST RNG/topology draws differ from GeNN',
        'power':PackageEnergy().metadata()})
    for source in sources:
        path=args.output/'source'/source.relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(source.read_bytes())
    control=execute(nest,args,0,True)
    write('diagnostic.json',control)
    results=[]
    for repetition in range(1,args.repetitions+1):
        result=execute(nest,args,repetition,False)
        for key in ['exc_spikes','inh_spikes']:
            if result[key]!=control[key]:raise RuntimeError(f'Diagnostic/timing mismatch: {key}')
        write(f'r{repetition}.json',result);results.append(result)
        print(f'TIMING rule={args.rule} threads={args.threads} r={repetition} us_per_step={result["us_per_step"]:.6f}',flush=True)
    times=[r['us_per_step'] for r in results]
    write('summary.json',{'rule':args.rule,'threads':args.threads,'median_us_per_step':float(np.median(times)),
        'range':[min(times),max(times)],'iqr':float(np.subtract(*np.percentile(times,[75,25]))),
        'repetitions':args.repetitions,'results':results,'diagnostic':control})
    print(f'COMPLETE output={args.output}',flush=True)


if __name__=='__main__':main()
