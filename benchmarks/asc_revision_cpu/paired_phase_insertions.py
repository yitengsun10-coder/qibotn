"""Fresh unseen-input timing for parity-phase insertion, with official baseline.

No old timing row is reused. A failed/partial run has no publishable ratio.
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
HELPERS = [Path(__file__), *[HERE / name for name in
           ("upstream_reference.py", "candidate_mixed_schedule.py", "candidate_phase_blocks.py",
            "candidate_phase_insertions.py", "evaluate.py", "probe_linear_holdout.py")]]


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def worker(args):
    import numpy as np
    import psutil
    from qibo.backends import NumpyBackend
    from qibotn.backends import quimb as module
    from threadpoolctl import threadpool_info, threadpool_limits
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_revision_cpu.candidate_mixed_schedule import optimize_mixed_schedule
    from benchmarks.asc_revision_cpu.candidate_phase_insertions import optimize_phase_insertions
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    from benchmarks.asc_revision_cpu.evaluate import observable, dense_expectation
    psutil.Process().cpu_affinity(args.affinity)
    circuit = build("mixed_entanglers", args.seed, args.qubits, args.layers)
    qasm = circuit.to_qasm()
    (args.output / "input.qasm").write_text(qasm, encoding="utf-8")
    klass, ref = load_reference(args.output)
    data = {"status": "preparing", "qubits": args.qubits, "layers": args.layers,
            "seed": args.seed, "gates": len(circuit.queue), "depth": circuit.depth,
            "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
            "precision": "complex128", "cutoff": 0, "max_bond": None,
            "threads": 1, "affinity": args.affinity, "warmup_repeats": 1,
            "measured_repeats": args.repeats, "reference": ref, "rows": [], "full_state": {},
            "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (*CORE, *HELPERS)},
            "scope": "fresh mixed-gate research holdout, not a formal 30q acceptance",
            "timing_scope": "conversion/compiler, all gates and original four ZZ terms; excludes imports and independent oracle"}
    path = args.output / "result.json"
    save(path, data)
    original = module.optimize_and_schedule
    with threadpool_limits(limits=1):
        data["threadpools"] = threadpool_info()
        baseline = klass()
        optimized = module.QuimbNumpyBackend()
        baseline.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
        optimized.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
        schedulers = {"mixed_schedule": optimize_mixed_schedule, "parity_insertions": optimize_phase_insertions}
        oracle = NumpyBackend().execute_circuit(circuit).state()
        for method, scheduler in schedulers.items():
            module.optimize_and_schedule = scheduler
            state, prep = optimized._prepare_optimized_mps(circuit)
            actual = np.asarray(state.to_dense()).reshape(-1)
            norm, overlap = float(np.vdot(actual, actual).real), np.vdot(oracle, actual)
            fidelity = float(abs(overlap) ** 2 / (norm * np.vdot(oracle, oracle).real))
            error = float(np.max(np.abs(actual * np.exp(-1j * np.angle(overlap)) - oracle)))
            data["full_state"][method] = {"norm_squared": norm, "fidelity": fidelity,
                "max_state_error": error, "max_bond_observed": state.max_bond(), "preparation": prep}
            save(path, data)
            if not all(map(math.isfinite, (norm, fidelity, error))) or abs(norm - 1) > 1e-10 or abs(fidelity - 1) > 1e-10 or error > 1e-9:
                raise RuntimeError("Independent full-state failure")
            del state, actual
        labels, sites, coefficients = observable(args.qubits, "zz4")
        exact = dense_expectation(oracle, args.qubits, labels, sites, coefficients)
        del oracle
        samples = {method: [] for method in ("upstream_typo_only", *schedulers)}
        for repeat in range(args.repeats + 1):
            methods = list(samples)
            if repeat % 2:
                methods.reverse()
            for method in methods:
                target = baseline if method == "upstream_typo_only" else optimized
                if method in schedulers:
                    module.optimize_and_schedule = schedulers[method]
                started = time.perf_counter()
                value = float(target.exp_value_observable_symbolic(circuit, labels, sites, coefficients, args.qubits))
                elapsed, error = time.perf_counter() - started, abs(value - exact)
                row = {"method": method, "repeat": repeat, "runtime_s": elapsed,
                       "value": value, "exact_value": exact, "abs_error": error}
                if method != "upstream_typo_only":
                    row["diagnostics"] = target.last_expectation_diagnostics
                data["rows"].append(row)
                data["status"] = "timing"
                save(path, data)
                print(json.dumps({k: row[k] for k in ("method", "repeat", "runtime_s", "abs_error")}), flush=True)
                if not all(map(math.isfinite, (value, elapsed, error))) or elapsed <= 0 or error > 1e-10:
                    raise RuntimeError("Observable failure")
                if repeat:
                    samples[method].append(elapsed)
        if circuit.to_qasm() != qasm:
            raise RuntimeError("Original input changed")
        means = {method: statistics.mean(values) for method, values in samples.items()}
        data["means_s"] = means
        data["stdev_s"] = {method: statistics.stdev(values) for method, values in samples.items()}
        data["speedups_vs_upstream"] = {method: means["upstream_typo_only"] / means[method] for method in schedulers}
        data["insertion_speedup_vs_mixed"] = means["mixed_schedule"] / means["parity_insertions"]
        data["status"] = "pass"
        module.optimize_and_schedule = original
        save(path, data)
        print(json.dumps({"status": "pass", "means_s": means,
                          "speedups_vs_upstream": data["speedups_vs_upstream"]}), flush=True)


def suite(args):
    import psutil
    if args.output.exists():
        raise RuntimeError("Fresh directory required")
    if psutil.virtual_memory().available < 1200 * 2**20:
        raise RuntimeError("Not started: system available memory below 1200 MiB")
    args.output.mkdir(parents=True)
    save(args.output / "run_metadata.json", {"started_at": datetime.now().astimezone().isoformat(),
        "controller_pid": os.getpid(), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "policy": "one unseen input, no retries, no old timing rows, no output-based choice"})
    for path in (*CORE, *HELPERS):
        shutil.copyfile(path, args.output / (path.stem + "_snapshot.py"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "QUIMB_NUM_PROCS"):
        env[key] = "1"
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
               "--qubits", str(args.qubits), "--layers", str(args.layers), "--seed", str(args.seed),
               "--repeats", str(args.repeats), "--affinity", *map(str, args.affinity)]
    started, peak, reason = time.monotonic(), 0, None
    with (args.output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
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
    report = {"status": "pass" if passed else "incomplete_or_failed", "exit_code": code,
              "stop_reason": reason, "elapsed_s": time.monotonic() - started,
              "peak_process_tree_rss_mib": peak / 2**20,
              "budgets": {"seconds": args.seconds, "memory_mib": args.memory_mib}}
    save(args.output / "summary.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--affinity", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--memory-mib", type=float, default=900)
    args = parser.parse_args()
    if args.qubits < 2 or args.layers < 1 or args.repeats < 3:
        parser.error("Need >=2 qubits, >=1 layer, >=3 measured repeats")
    worker(args) if args.worker else suite(args)
