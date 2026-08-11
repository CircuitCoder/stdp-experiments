#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ports.brian2_port import Brian2Brunel
from ports.common import (
    DT_MS,
    base_manifest,
    create_output,
    make_model,
    stdp_post_path_delay_ms,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Brian2CUDA standalone source for a Brunel STDP port"
    )
    parser.add_argument("--rule", choices=("additive", "morrison"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--network-scale", type=float, default=0.002)
    parser.add_argument("--indegree-scale", type=float, default=0.002)
    parser.add_argument("--sim-ms", type=float, default=1.0)
    args = parser.parse_args()

    import brian2 as b
    import brian2cuda

    model = make_model(args.rule, args.network_scale, args.indegree_scale)
    manifest = base_manifest("brian2cuda-codegen", model)
    manifest.update(
        {
            "brian2_version": getattr(b, "__version__", "unknown"),
            "brian2cuda_version": getattr(brian2cuda, "__version__", "unknown"),
            "compile": False,
            "run": False,
            "scope": "offline source-generation validation",
            "sim_ms": args.sim_ms,
            "stdp_timing": "nest_dendritic",
            "stdp_tie_order": "nest_causal_boundary",
            "stdp_post_path_delay_ms": stdp_post_path_delay_ms("nest_dendritic"),
        }
    )
    create_output(args.output, manifest)
    b.prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    b.prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    b.prefs.devices.cuda_standalone.cuda_backend.compute_capability = 8.0
    b.prefs.devices.cuda_standalone.cuda_backend.cuda_runtime_version = 12.0
    b.set_device("cuda_standalone", build_on_run=False)
    network = Brian2Brunel(
        model=model,
        seed=args.seed,
        state_seed=None,
        n_record=min(1000, model.ne),
        codegen_target="cython",
        connectivity_target_chunk=32,
        stdp_timing="nest_dendritic",
        stdp_tie_order="nest_causal_boundary",
    )
    network.network.run(args.sim_ms * b.ms, namespace={})
    code_directory = args.output / "cuda_standalone"
    device = b.get_device()
    device.generate_makefile = lambda *unused_args, **unused_kwargs: None
    device.build(
        directory=str(code_directory),
        compile=False,
        run=False,
        clean=True,
    )
    generated = sorted(path.name for path in code_directory.rglob("*") if path.is_file())
    with (args.output / "codegen.json").open("x", encoding="ascii") as stream:
        json.dump(
            {
                "generated_file_count": len(generated),
                "generated_files": generated,
                "simulated_steps_scheduled": round(args.sim_ms / DT_MS),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(
        f"CODEGEN_COMPLETE backend=brian2cuda rule={args.rule} "
        f"files={len(generated)} compile=false run=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
