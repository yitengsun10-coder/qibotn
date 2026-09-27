#!/usr/bin/env python3
"""Scan large-workload MPS fidelity without repeating timing measurements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark import VARIANTS, Workload, build_circuit, validate_mps_fidelity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        required=True,
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimizer", default="auto-hq")
    parser.add_argument("--min-fidelity", type=float, default=0.999)
    args = parser.parse_args()

    if "baseline" not in args.variant:
        parser.error("baseline must be included")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["QUIMB_NUM_PROCS"] = str(args.threads)

    circuit = build_circuit(Workload(args.qubits, args.layers))
    print(
        f"FIDELITY_SCAN qubits={args.qubits} layers={args.layers} "
        f"gates={len(circuit.queue)} depth={circuit.depth}",
        flush=True,
    )
    results = validate_mps_fidelity(
        circuit,
        variant_names=args.variant,
        threads=args.threads,
        contraction_optimizer=args.optimizer,
    )
    payload = {
        "qubits": args.qubits,
        "layers": args.layers,
        "gates": len(circuit.queue),
        "depth": circuit.depth,
        "min_fidelity": args.min_fidelity,
        "variants": results,
    }
    for values in payload["variants"].values():
        values["fidelity_pass"] = (
            values["fidelity_vs_baseline"] >= args.min_fidelity
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
