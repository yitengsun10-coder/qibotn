#!/usr/bin/env python3
"""Run each CPU thread configuration in a fresh process and compare results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--optimizer", default="auto-hq")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    if 1 not in args.threads:
        parser.error("thread sweep must include the 1-thread baseline")
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).with_name("benchmark.py")
    workload_key = f"q{args.qubits}_l{args.layers}"
    collected = []

    for threads in args.threads:
        run_output = args.output / f"threads_{threads}"
        command = [
            sys.executable,
            str(benchmark),
            "--output",
            str(run_output),
            "--workload",
            f"{args.qubits}:{args.layers}",
            "--variant",
            "baseline",
            "--repeats",
            str(args.repeats),
            "--threads",
            str(threads),
            "--optimizer",
            args.optimizer,
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "NUMEXPR_NUM_THREADS": str(threads),
                "QUIMB_NUM_PROCS": str(threads),
            }
        )
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_output / "run.log").write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, flush=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"thread={threads} failed with exit code {completed.returncode}"
            )
        summary = json.loads((run_output / "summary.json").read_text("utf-8"))
        values = summary[workload_key]["variants"]["baseline"]
        collected.append(
            {
                "threads": threads,
                "steady_mean_runtime_s": values["steady_mean_runtime_s"],
                "steady_median_runtime_s": values["steady_median_runtime_s"],
                "steady_stdev_runtime_s": values["steady_stdev_runtime_s"],
                "cold_runtime_s": values["cold_runtime_s"],
                "peak_rss_mib": values["peak_rss_mib"],
                "expectation": values["mean_expectation"],
                "qasm_sha256": summary[workload_key]["qasm_sha256"],
            }
        )

    baseline = next(row for row in collected if row["threads"] == 1)
    for row in collected:
        row["speedup_vs_1_thread"] = (
            baseline["steady_mean_runtime_s"] / row["steady_mean_runtime_s"]
        )
        row["abs_error_vs_1_thread"] = abs(
            row["expectation"] - baseline["expectation"]
        )
        row["correct_within_tolerance"] = (
            row["abs_error_vs_1_thread"] <= args.tolerance
        )

    with (args.output / "thread_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(collected[0]))
        writer.writeheader()
        writer.writerows(collected)
    payload = {
        "qubits": args.qubits,
        "layers": args.layers,
        "repeats": args.repeats,
        "optimizer": args.optimizer,
        "tolerance": args.tolerance,
        "fixed_numerical_configuration": {
            "cutoff": 1e-12,
            "max_bond": 128,
            "variant": "baseline",
        },
        "results": collected,
    }
    (args.output / "thread_sweep_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if not all(row["correct_within_tolerance"] for row in collected):
        raise SystemExit("thread sweep changed the numerical output")


if __name__ == "__main__":
    main()
