"""Fresh matched-thread official-reference / frozen / candidate comparison.

One predetermined random mixed circuit, one warmup and three measured repeats.
The denominator is pinned upstream plus one attribute typo repair, not the
norm-guarded development reference. Partial rows never produce a speedup.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CORE = [ROOT / "src/qibotn/backends" / name for name in
        ("quimb.py", "_exact_numpy.py", "_linear_cnot.py", "_structured_numpy.py", "_cnot_schedule.py")]
HELPERS = [Path(__file__), HERE / "upstream_reference.py", HERE / "candidate_mixed_schedule.py",
           HERE / "candidate_phase_blocks.py", HERE / "evaluate.py", HERE / "probe_linear_holdout.py"]


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def worker(args):
    import numpy as np
    import psutil
    from qibo.backends import NumpyBackend
    from qibotn.backends import quimb as module
    from threadpoolctl import threadpool_info, threadpool_limits
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_revision_cpu.candidate_mixed_schedule import optimize_mixed_schedule
    from benchmarks.asc_revision_cpu.candidate_phase_blocks import optimize_phase_blocks
    from benchmarks.asc_revision_cpu.evaluate import dense_expectation, observable
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    process = psutil.Process()
    process.cpu_affinity(args.affinity)
    circuit = build("mixed_entanglers", args.seed, n=args.qubits, layers=args.layers)
    qasm = circuit.to_qasm()
    (args.output / "input.qasm").write_text(qasm, encoding="utf-8")
    upstream_class, reference_metadata = load_reference(args.output)
    counts = {}
    for gate in circuit.queue:
        name = type(gate).__name__
        counts[name] = counts.get(name, 0) + 1
    result = {
        "status": "preparing", "qubits": args.qubits, "layers": args.layers,
        "seed": args.seed, "gates": len(circuit.queue), "depth": circuit.depth,
        "gate_counts": counts, "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "precision": "complex128", "cutoff": 0, "max_bond": None,
        "warmup_repeats": 1, "measured_repeats": args.repeats,
        "affinity": args.affinity, "threads": 1,
        "reference": reference_metadata, "rows": [], "full_state": {},
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (*CORE, *HELPERS)},
        "scope": "new supplementary mixed-gate test; not a 30-qubit acceptance result",
        "timing_scope": "fresh circuit conversion/compiler, all gates and the same original four ZZ terms; excludes imports, backend initialization and independent correctness oracle",
        "optimized_normalization": "optimized norm check/normalization is timed; official reference retains its original raw output; exact unitary state norms must be one within 1e-10",
    }
    result_path = args.output / "result.json"
    save(result_path, result)
    original_scheduler = module.optimize_and_schedule
    with threadpool_limits(limits=1):
        result["threadpools"] = threadpool_info()
        optimized = module.QuimbNumpyBackend()
        optimized.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
        baseline = upstream_class()
        baseline.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
        oracle = NumpyBackend().execute_circuit(circuit).state()
        schedulers = {"frozen": original_scheduler, "candidate": optimize_mixed_schedule,
                      "phase_candidate": optimize_phase_blocks}
        for method, scheduler in schedulers.items():
            module.optimize_and_schedule = scheduler
            state, preparation = optimized._prepare_optimized_mps(circuit)
            actual = np.asarray(state.to_dense()).reshape(-1)
            norm = float(np.vdot(actual, actual).real)
            overlap = np.vdot(oracle, actual)
            fidelity = float(abs(overlap) ** 2 / (norm * np.vdot(oracle, oracle).real))
            error = float(np.max(np.abs(actual * np.exp(-1j * np.angle(overlap)) - oracle)))
            result["full_state"][method] = {
                "norm_squared": norm, "fidelity": fidelity, "max_state_error": error,
                "max_bond_observed": state.max_bond(), "preparation": preparation,
            }
            save(result_path, result)
            if (not all(map(math.isfinite, (norm, fidelity, error))) or
                    abs(norm - 1) > 1e-10 or abs(fidelity - 1) > 1e-10 or error > 1e-9):
                raise RuntimeError(f"{method}: independent full-state validation failed")
            del state, actual
        labels, sites, coefficients = observable(args.qubits, "zz4")
        exact = dense_expectation(oracle, args.qubits, labels, sites, coefficients)
        del oracle
        samples = {method: [] for method in ("upstream_typo_only", *schedulers)}
        for repeat in range(args.repeats + 1):
            methods = list(samples)
            # A/B/C then C/B/A: no choice based on observed performance.
            if repeat % 2:
                methods.reverse()
            for method in methods:
                if method == "upstream_typo_only":
                    target = baseline
                else:
                    target = optimized
                    module.optimize_and_schedule = schedulers[method]
                started = time.perf_counter()
                value = float(target.exp_value_observable_symbolic(
                    circuit, labels, sites, coefficients, args.qubits))
                elapsed = time.perf_counter() - started
                error = abs(value - exact)
                row = {"method": method, "repeat": repeat, "runtime_s": elapsed,
                       "value": value, "exact_value": exact, "abs_error": error}
                if method != "upstream_typo_only":
                    row["diagnostics"] = target.last_expectation_diagnostics
                result["rows"].append(row)
                result["status"] = "timing"
                save(result_path, result)
                print(json.dumps({k: row[k] for k in ("method", "repeat", "runtime_s", "abs_error")}), flush=True)
                if not all(map(math.isfinite, (value, elapsed, error))) or elapsed <= 0 or error > 1e-10:
                    raise RuntimeError(f"{method}: observable validation failed")
                if repeat:
                    samples[method].append(elapsed)
        if circuit.to_qasm() != qasm:
            raise RuntimeError("Original input circuit was mutated")
        result["means_s"] = {method: statistics.mean(values) for method, values in samples.items()}
        result["stdev_s"] = {method: statistics.stdev(values) for method, values in samples.items()}
        means = result["means_s"]
        result["speedups"] = {
            "frozen_vs_upstream_typo_only": means["upstream_typo_only"] / means["frozen"],
            "candidate_vs_upstream_typo_only": means["upstream_typo_only"] / means["candidate"],
            "candidate_vs_frozen": means["frozen"] / means["candidate"],
            "phase_candidate_vs_upstream_typo_only": means["upstream_typo_only"] / means["phase_candidate"],
            "phase_candidate_vs_frozen": means["frozen"] / means["phase_candidate"],
        }
        result["status"] = "pass"
        module.optimize_and_schedule = original_scheduler
        save(result_path, result)
        print(json.dumps({"status": "pass", "means_s": means, "speedups": result["speedups"]}), flush=True)


def suite(args):
    import psutil
    if args.output.exists():
        raise RuntimeError("Fresh evidence directory required")
    if psutil.virtual_memory().available < 1200 * 2**20:
        raise RuntimeError("Less than 1200 MiB available: no test started")
    args.output.mkdir(parents=True)
    save(args.output / "run_metadata.json", {
        "started_at": datetime.now().astimezone().isoformat(), "controller_pid": os.getpid(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "policy": "one predeclared input, no retries or output-dependent candidate selection",
    })
    for path in (*CORE, *HELPERS):
        shutil.copyfile(path, args.output / (path.stem + "_snapshot.py"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "QUIMB_NUM_PROCS"):
        env[key] = "1"
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker",
               "--output", str(args.output), "--qubits", str(args.qubits), "--layers", str(args.layers),
               "--seed", str(args.seed), "--repeats", str(args.repeats), "--affinity", *map(str, args.affinity)]
    started, peak, reason = time.monotonic(), 0, None
    with (args.output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                tree = psutil.Process(process.pid)
                peak = max(peak, tree.memory_info().rss + sum(c.memory_info().rss for c in tree.children(recursive=True)))
                if peak > args.memory_mib * 2**20:
                    reason = "process_tree_memory_budget_exceeded"
                elif psutil.virtual_memory().available < 800 * 2**20:
                    reason = "system_available_memory_below_800_MiB"
                elif time.monotonic() - started > args.seconds:
                    reason = "wall_clock_budget_exceeded"
                if reason:
                    for child in tree.children(recursive=True):
                        child.kill()
                    process.kill()
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.1)
        code = process.wait()
    path = args.output / "result.json"
    passed = code == 0 and reason is None and path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    summary = {"status": "pass" if passed else "incomplete_or_failed", "exit_code": code,
               "stop_reason": reason, "elapsed_s": time.monotonic() - started,
               "peak_process_tree_rss_mib": peak / 2**20,
               "budgets": {"seconds": args.seconds, "memory_mib": args.memory_mib}}
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261091)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--affinity", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--memory-mib", type=float, default=900)
    parser.add_argument("--seconds", type=float, default=600)
    arguments = parser.parse_args()
    if arguments.qubits < 2 or arguments.layers < 1 or arguments.repeats < 2:
        parser.error("Need >=2 qubits, >=1 layer and >=2 measured repeats")
    worker(arguments) if arguments.worker else suite(arguments)
