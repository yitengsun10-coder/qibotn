"""Allocation-free upper-bound planning for exact open-boundary qubit MPS.

No circuit simulation, performance prediction or hardware purchase is made.
Full Schmidt-rank dimensions can overestimate a low-entanglement circuit; the
bound is important precisely to avoid treating such a special case as general.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import psutil


def exact_mps_bounds(qubits, itemsize=16):
    if qubits < 2 or itemsize < 1:
        raise ValueError("Need at least two qubits and a positive item size")
    bonds = [1 << min(k, qubits - k) for k in range(qubits + 1)]
    state_bytes = (1 << qubits) * itemsize
    tensor_bytes = sum(2 * bonds[k] * bonds[k + 1] for k in range(qubits)) * itemsize
    merged_bytes = max(4 * bonds[k] * bonds[k + 2] for k in range(qubits - 1)) * itemsize
    return {"qubits": qubits, "precision": "complex128" if itemsize == 16 else "unspecified",
        "bond_dimension_upper_bound": max(bonds), "dense_state_bytes": state_bytes,
        "full_rank_mps_arrays_bytes": tensor_bytes, "largest_merged_tensor_bytes": merged_bytes,
        "mps_plus_one_merged_bytes": tensor_bytes + merged_bytes,
        "excludes": ["QR/SVD workspaces", "identity-gauge snapshots", "gate/output copies",
                     "Python/dependency overhead", "other applications"],
        "scope": "Dimension-derived full-rank upper bound, not observed allocation or measured runtime"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, nargs="+", default=[18, 20, 24, 30])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    memory = psutil.virtual_memory()
    data = {"recorded_at": datetime.now().astimezone().isoformat(),
        "host": {"physical_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
                 "total_memory_bytes": memory.total, "available_memory_bytes": memory.available},
        "bounds": [exact_mps_bounds(n) for n in args.qubits],
        "simulation_performed": False, "cloud_resources_started": False,
        "acceptance_or_7x_proven": False,
        "note": "A larger-memory machine would permit broader tests, not guarantee their runtime, speedup, or competition acceptance."}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, args.output)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
