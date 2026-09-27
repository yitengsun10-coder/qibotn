#!/usr/bin/env python3
"""Reproducible large-workload benchmark for the QiboTN Quimb CPU path.

The benchmark keeps the circuit, observable, backend, precision, and output
requirements explicit and records every changed parameter. Numerical variants
are treated as candidates until an MPS-overlap check accepts them. The formal
thread experiment keeps the numerical configuration fixed and changes only the
CPU thread pool in fresh processes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
from qibo import Circuit, gates
from qibotn.backends.quimb import QuimbNumpyBackend
from threadpoolctl import threadpool_info, threadpool_limits


@dataclass(frozen=True)
class Workload:
    qubits: int
    layers: int


@dataclass(frozen=True)
class Variant:
    name: str
    cutoff: float
    max_bond: int


PROFILES = {
    "quick": [Workload(12, 5)],
    "tune": [Workload(25, 67)],
    "formal": [Workload(30, 112)],
    "scale": [Workload(20, 33), Workload(25, 67), Workload(30, 112)],
}

VARIANTS = {
    "baseline": Variant("baseline", cutoff=1e-12, max_bond=128),
    "cutoff_1e10_b128": Variant(
        "cutoff_1e10_b128", cutoff=1e-10, max_bond=128
    ),
    "cutoff_1e8_b128": Variant(
        "cutoff_1e8_b128", cutoff=1e-8, max_bond=128
    ),
    "cutoff_1e7_b128": Variant(
        "cutoff_1e7_b128", cutoff=1e-7, max_bond=128
    ),
    "cutoff_1e6_b128": Variant(
        "cutoff_1e6_b128", cutoff=1e-6, max_bond=128
    ),
    "cutoff_1e5_b128": Variant(
        "cutoff_1e5_b128", cutoff=1e-5, max_bond=128
    ),
    "cutoff_1e10_b96": Variant(
        "cutoff_1e10_b96", cutoff=1e-10, max_bond=96
    ),
    "cutoff_1e8_b96": Variant("cutoff_1e8_b96", cutoff=1e-8, max_bond=96),
    "cutoff_1e8_b64": Variant("cutoff_1e8_b64", cutoff=1e-8, max_bond=64),
    "cutoff_1e6_b64": Variant("cutoff_1e6_b64", cutoff=1e-6, max_bond=64),
}

DEFAULT_VARIANTS = {
    "quick": ["baseline", "cutoff_1e8_b64"],
    "tune": list(VARIANTS),
    # Relaxed cutoffs failed the 30-qubit MPS-fidelity threshold. The formal
    # numerical profile therefore remains conservative; thread_sweep.py owns
    # the accepted performance comparison.
    "formal": ["baseline"],
    "scale": ["baseline", "cutoff_1e8_b128"],
}


class PeakRSSSampler:
    """Sample resident memory of this process and all child processes."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _rss(self) -> int:
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval_s)
        self.peak_bytes = max(self.peak_bytes, self._rss())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()


def build_circuit(workload: Workload) -> Circuit:
    """Create a deterministic, hardware-efficient, entangling circuit.

    For 30 qubits and 112 layers this has 9,998 gates and a depth of roughly
    450, which satisfies the requested large-workload range without changing
    the circuit between variants.
    """

    n = workload.qubits
    circuit = Circuit(n)
    for q in range(n):
        circuit.add(gates.H(q))

    for layer in range(workload.layers):
        for q in range(n):
            angle = 0.013 * (layer + 1) * (q + 1)
            circuit.add(gates.RX(q, theta=angle))
            circuit.add(gates.RY(q, theta=0.7 * angle + 0.11))
        parity = layer % 2
        for q in range(parity, n - 1, 2):
            circuit.add(gates.CNOT(q, q + 1))
        for q in range(1 - parity, n - 1, 2):
            circuit.add(gates.CNOT(q, q + 1))
    return circuit


def circuit_sha256(circuit: Circuit) -> str:
    return hashlib.sha256(circuit.to_qasm().encode("utf-8")).hexdigest()


def make_observable(nqubits: int):
    """Return the same four local ZZ terms for every tested variant."""

    pairs = sorted(
        {
            (0, 1),
            (nqubits // 3, nqubits // 3 + 1),
            (2 * nqubits // 3, 2 * nqubits // 3 + 1),
            (nqubits - 2, nqubits - 1),
        }
    )
    return ["zz"] * len(pairs), pairs, [1.0 / len(pairs)] * len(pairs)


def run_once(
    circuit: Circuit,
    variant: Variant,
    threads: int,
    contraction_optimizer: str,
) -> dict:
    operators, sites, coeffs = make_observable(circuit.nqubits)
    backend = QuimbNumpyBackend(contraction_optimizer=contraction_optimizer)
    backend.configure_tn_simulation(
        ansatz="mps",
        max_bond_dimension=variant.max_bond,
        svd_cutoff=variant.cutoff,
    )

    with threadpool_limits(limits=threads):
        with PeakRSSSampler() as memory:
            started = time.perf_counter()
            value = backend.exp_value_observable_symbolic(
                circuit,
                operators_list=operators,
                sites_list=sites,
                coeffs_list=coeffs,
                nqubits=circuit.nqubits,
            )
            elapsed = time.perf_counter() - started

    return {
        "runtime_s": elapsed,
        "peak_rss_mib": memory.peak_bytes / 2**20,
        "expectation": float(np.real(value)),
        "finite": bool(np.isfinite(value)),
    }


def build_mps_state(
    circuit: Circuit,
    variant: Variant,
    threads: int,
    contraction_optimizer: str,
):
    """Build the QiboTN Quimb CPU MPS without materializing a dense vector."""

    backend = QuimbNumpyBackend(contraction_optimizer=contraction_optimizer)
    backend.configure_tn_simulation(
        ansatz="mps",
        max_bond_dimension=variant.max_bond,
        svd_cutoff=variant.cutoff,
    )
    with threadpool_limits(limits=threads):
        started = time.perf_counter()
        state = backend._qibo_circuit_to_quimb(
            circuit,
            quimb_circuit_type=backend.circuit_ansatz,
            gate_opts={
                "max_bond": variant.max_bond,
                "cutoff": variant.cutoff,
            },
        ).psi
        elapsed = time.perf_counter() - started
    return state, elapsed


def validate_mps_fidelity(
    circuit: Circuit,
    variant_names: list[str],
    threads: int,
    contraction_optimizer: str,
) -> dict:
    """Compare each optimized MPS with the conservative baseline MPS."""

    baseline, baseline_build_s = build_mps_state(
        circuit,
        VARIANTS["baseline"],
        threads=threads,
        contraction_optimizer=contraction_optimizer,
    )
    baseline_norm = baseline.H @ baseline
    results = {
        "baseline": {
            "fidelity_vs_baseline": 1.0,
            "build_runtime_s": baseline_build_s,
            "max_bond_observed": baseline.max_bond(),
            "norm": float(np.real(baseline_norm)),
        }
    }
    for variant_name in variant_names:
        if variant_name == "baseline":
            continue
        state, build_s = build_mps_state(
            circuit,
            VARIANTS[variant_name],
            threads=threads,
            contraction_optimizer=contraction_optimizer,
        )
        started = time.perf_counter()
        overlap = baseline.H @ state
        state_norm = state.H @ state
        fidelity = float(
            abs(overlap) ** 2 / (np.real(baseline_norm) * np.real(state_norm))
        )
        results[variant_name] = {
            "fidelity_vs_baseline": fidelity,
            "build_runtime_s": build_s,
            "overlap_runtime_s": time.perf_counter() - started,
            "max_bond_observed": state.max_bond(),
            "norm": float(np.real(state_norm)),
        }
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], tolerance: float) -> dict:
    summary: dict[str, dict] = {}
    workload_keys = sorted({(row["qubits"], row["layers"]) for row in rows})
    for qubits, layers in workload_keys:
        selected_workload = [
            row
            for row in rows
            if row["qubits"] == qubits and row["layers"] == layers
        ]
        baseline_rows = [
            row for row in selected_workload if row["variant"] == "baseline"
        ]
        if not baseline_rows:
            continue
        baseline_expectation = statistics.mean(
            row["expectation"] for row in baseline_rows
        )
        baseline_steady = [
            row["runtime_s"] for row in baseline_rows if row["repeat"] > 1
        ] or [row["runtime_s"] for row in baseline_rows]
        baseline_runtime = statistics.mean(baseline_steady)

        key = f"q{qubits}_l{layers}"
        summary[key] = {
            "qubits": qubits,
            "layers": layers,
            "gates": selected_workload[0]["gates"],
            "depth": selected_workload[0]["depth"],
            "dense_state_gib": selected_workload[0]["dense_state_gib"],
            "qasm_sha256": selected_workload[0]["qasm_sha256"],
            "variants": {},
        }
        for variant_name in sorted({row["variant"] for row in selected_workload}):
            variant_rows = [
                row for row in selected_workload if row["variant"] == variant_name
            ]
            steady_rows = [row for row in variant_rows if row["repeat"] > 1]
            if not steady_rows:
                steady_rows = variant_rows
            runtimes = [row["runtime_s"] for row in steady_rows]
            expectations = [row["expectation"] for row in variant_rows]
            max_abs_error = max(
                abs(value - baseline_expectation) for value in expectations
            )
            mean_runtime = statistics.mean(runtimes)
            summary[key]["variants"][variant_name] = {
                "steady_mean_runtime_s": mean_runtime,
                "steady_median_runtime_s": statistics.median(runtimes),
                "steady_stdev_runtime_s": (
                    statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
                ),
                "cold_runtime_s": variant_rows[0]["runtime_s"],
                "peak_rss_mib": max(row["peak_rss_mib"] for row in variant_rows),
                "mean_expectation": statistics.mean(expectations),
                "max_abs_error_vs_baseline": max_abs_error,
                "correct_within_tolerance": max_abs_error <= tolerance,
                "speedup_vs_baseline": baseline_runtime / mean_runtime,
                "all_finite": all(row["finite"] for row in variant_rows),
            }
    return summary


def parse_workload(value: str) -> Workload:
    try:
        qubits_text, layers_text = value.split(":", maxsplit=1)
        return Workload(int(qubits_text), int(layers_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("workload must use QUBITS:LAYERS") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="quick"
    )
    parser.add_argument(
        "--workload",
        action="append",
        type=parse_workload,
        help="override the profile with QUBITS:LAYERS; may be repeated",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="variant to run; may be repeated",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimizer", default="auto-hq")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--fidelity-check", action="store_true")
    parser.add_argument("--min-fidelity", type=float, default=0.999)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["QUIMB_NUM_PROCS"] = str(args.threads)

    workloads = args.workload or PROFILES[args.profile]
    variant_names = args.variant or DEFAULT_VARIANTS[args.profile]
    if "baseline" not in variant_names:
        parser.error("baseline must be included for a valid speedup comparison")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_gib": psutil.virtual_memory().total / 2**30,
        "profile": "custom" if args.workload else args.profile,
        "workloads": [asdict(workload) for workload in workloads],
        "variants": [asdict(VARIANTS[name]) for name in variant_names],
        "threads": args.threads,
        "optimizer": args.optimizer,
        "tolerance": args.tolerance,
        "min_fidelity": args.min_fidelity,
        "backend": "QiboTN Quimb NumPy CPU",
        "timing_scope": "official QiboTN circuit conversion, MPS gate application, and four-term ZZ expectation",
        "framework_capability": "QiboTN Quimb CPU MPS execution",
        "submitted_change": (
            "fixed Quimb expectation optimizer attribute; benchmarked MPS "
            "accuracy candidates; formal optimization controls CPU threads"
        ),
        "threadpools": threadpool_info(),
    }
    (args.output / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fidelity_results: dict[str, dict] = {}
    for workload in workloads:
        circuit = build_circuit(workload)
        qasm_hash = circuit_sha256(circuit)
        dense_state_gib = (2**workload.qubits * 16) / 2**30
        for variant_name in variant_names:
            variant = VARIANTS[variant_name]
            for repeat in range(1, args.repeats + 1):
                print(
                    f"START q={workload.qubits} layers={workload.layers} "
                    f"gates={len(circuit.queue)} depth={circuit.depth} "
                    f"variant={variant.name} repeat={repeat}",
                    flush=True,
                )
                started_utc = datetime.now(timezone.utc).isoformat()
                result = run_once(
                    circuit,
                    variant,
                    threads=args.threads,
                    contraction_optimizer=args.optimizer,
                )
                row = {
                    "profile": args.profile,
                    "qubits": workload.qubits,
                    "layers": workload.layers,
                    "gates": len(circuit.queue),
                    "depth": circuit.depth,
                    "dense_state_gib": dense_state_gib,
                    "qasm_sha256": qasm_hash,
                    "variant": variant.name,
                    "cutoff": variant.cutoff,
                    "max_bond": variant.max_bond,
                    "threads": args.threads,
                    "repeat": repeat,
                    "started_utc": started_utc,
                    **result,
                }
                rows.append(row)
                write_csv(args.output / "results.csv", rows)
                current_summary = summarize(rows, args.tolerance)
                (args.output / "summary.json").write_text(
                    json.dumps(current_summary, indent=2), encoding="utf-8"
                )
                print("DONE", json.dumps(row, ensure_ascii=False), flush=True)

        if args.fidelity_check:
            print(
                f"FIDELITY q={workload.qubits} layers={workload.layers}",
                flush=True,
            )
            key = f"q{workload.qubits}_l{workload.layers}"
            fidelity_results[key] = validate_mps_fidelity(
                circuit,
                variant_names=variant_names,
                threads=args.threads,
                contraction_optimizer=args.optimizer,
            )
            (args.output / "fidelity.json").write_text(
                json.dumps(fidelity_results, indent=2), encoding="utf-8"
            )

    final_summary = summarize(rows, args.tolerance)
    for workload_key, variants in fidelity_results.items():
        for variant_name, values in variants.items():
            if workload_key in final_summary:
                final_summary[workload_key]["variants"][variant_name].update(values)
                final_summary[workload_key]["variants"][variant_name][
                    "fidelity_pass"
                ] = values["fidelity_vs_baseline"] >= args.min_fidelity
    (args.output / "summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    failed = [
        (workload, variant)
        for workload, data in final_summary.items()
        for variant, values in data["variants"].items()
        if not values["correct_within_tolerance"] or not values["all_finite"]
    ]
    failed_fidelity = [
        (workload, variant)
        for workload, data in final_summary.items()
        for variant, values in data["variants"].items()
        if args.fidelity_check and not values.get("fidelity_pass", False)
    ]
    print(json.dumps(final_summary, indent=2), flush=True)
    if failed or failed_fidelity:
        raise SystemExit(
            f"correctness validation failed: observable={failed}, "
            f"fidelity={failed_fidelity}"
        )


if __name__ == "__main__":
    main()
