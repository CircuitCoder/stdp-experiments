#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zd3.constants import MODEL
from zd3.io import load_reference_triplets, save_checkpoint, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an immutable Brian 1 weight/theta pair to the portable ZD3 format."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--accepted-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = load_reference_triplets(args.weights, MODEL.n_input, MODEL.n_exc)
    theta_mv = np.asarray(np.load(args.theta, allow_pickle=False), dtype=np.float64) * 1000.0
    save_checkpoint(
        args.output,
        weights=weights,
        theta_mv=theta_mv,
        accepted_samples=args.accepted_samples,
        manifest={
            "backend": "brian1-import",
            "source_weights": str(args.weights),
            "source_weights_sha256": sha256_file(args.weights),
            "source_theta": str(args.theta),
            "source_theta_sha256": sha256_file(args.theta),
            "runtime_state_scope": "weights-and-theta-only",
            "theta_source_units": "volt",
        },
    )
    print(f"CONVERT_COMPLETE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
