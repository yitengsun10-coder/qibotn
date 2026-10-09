"""Budgeted multi-topology CPU validation and matched-thread performance audit.

This does not promote local validation to a 30-qubit competition result.
The old thread-only speedup is intentionally not used as an algorithm baseline.
Each worker uses fresh QiboTN state preparation for every timed call. Exact
gate preparation and observable reuse are changed only on the optimized path.
Independent NumPy state-vector results are used
for correctness, never for the performance denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def build_circuit(topology, n, layers, seed):
    import numpy as np
    from qibo import Circuit, gates
    rng = np.random.default_rng(seed)
    circuit = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            circuit.add(gates.RY(q, theta=float(rng.uniform(-np.pi, np.pi))))
            circuit.add(gates.RZ(q, theta=float(rng.uniform(-np.pi, np.pi))))
        if topology == "chain":
            edges = [(q, q + 1) for q in range(n - 1)]
        elif topology == "ring":
            edges = [(q, (q + 1) % n) for q in range(n)]
        elif topology == "long_range":
            permutation = rng.permutation(n)
            edges = [(int(q), int(r)) for q, r in zip(permutation[::2], permutation[1::2])]
        elif topology == "all_to_all":
            edges = [(q, r) for q in range(n) for r in range(q + 1, n)]
        else:
            raise ValueError(topology)
        for q, r in edges:
            circuit.add(gates.CNOT(q, r))
    return circuit


def observable(n, family):
    if family == "zz4":
        sites = [(0, 1), (n // 3, n // 3 + 1), (2 * n // 3, 2 * n // 3 + 1), (n - 2, n - 1)]
        return ["zz"] * 4, sites, [0.25] * 4
    labels, sites = [], []
    for q in range(n):
        for label in "xyz":
            labels.append(label)
            sites.append((q,))
    for q in range(n):
        for r in range(q + 1, n):
            labels.append("zz" if (q + r) % 2 else "xy")
            sites.append((q, r))
    labels.extend(["xyz", "zyx", "i"])
    sites.extend([(0, n // 2, n - 1), (n - 1, n // 2, 0), (n // 2,)])
    coefficients = [(-1.0 if k % 3 == 0 else 1.0) / len(labels) for k in range(len(labels))]
    return labels, sites, coefficients


def dense_expectation(state, n, labels, sites, coefficients):
    import numpy as np
    # Explicit Pauli matrices: independent of Quimb's observable evaluator.
    matrices = {"i": np.eye(2), "x": np.array([[0, 1], [1, 0]]),
                "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1])}
    values = []
    for opstr, where in zip(labels, sites):
        transformed = state.reshape((2,) * n)
        for label, site in zip(opstr, where):
            transformed = np.moveaxis(np.tensordot(matrices[label], transformed, axes=(1, site)), 0, site)
        values.append(float(np.vdot(state, transformed.reshape(-1)).real))
    return float(np.dot(coefficients, values) / np.vdot(state, state).real)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def worker(args):
    import psutil
    process = psutil.Process()
    if args.affinity_cpu is not None:
        if args.affinity_cpu not in process.cpu_affinity():
            raise ValueError("Requested logical CPU is not available to the worker")
        process.cpu_affinity([args.affinity_cpu])
    elif args.affinity_cpus is not None:
        if not set(args.affinity_cpus) <= set(process.cpu_affinity()):
            raise ValueError("Requested logical CPUs are not available")
        process.cpu_affinity(args.affinity_cpus)
    import importlib.metadata
    import numpy as np
    from qibo.backends import NumpyBackend
    from qibotn.backends.quimb import QuimbNumpyBackend
    from threadpoolctl import threadpool_info, threadpool_limits

    args.output.mkdir(parents=True, exist_ok=True)
    circuit = build_circuit(args.topology, args.qubits, args.layers, args.seed)
    qasm_hash = hashlib.sha256(circuit.to_qasm().encode()).hexdigest()
    backend = QuimbNumpyBackend(contraction_optimizer="auto-hq")
    # Exact MPS, not a fixed 128-bond approximation. Resource limits are enforced
    # by the parent. Large entangled states must fail the budget, not silently
    # lower numerical accuracy to obtain a fast result.
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0.0)
    with threadpool_limits(limits=args.threads):
        reference_state = NumpyBackend().execute_circuit(circuit).state()
        mps, preparation = backend._prepare_optimized_mps(circuit)
        mps_vector = np.asarray(mps.to_dense()).reshape(-1)
        mps_norm = float(np.vdot(mps_vector, mps_vector).real)
        reference_norm = float(np.vdot(reference_state, reference_state).real)
        fidelity = float(abs(np.vdot(reference_state, mps_vector))**2 / (mps_norm * reference_norm))
        phase = np.vdot(reference_state, mps_vector)
        aligned = mps_vector * np.exp(-1j * np.angle(phase))
        state_error = float(np.max(np.abs(aligned - reference_state)))
        state_pass = fidelity >= 1.0 - 1e-10 and abs(mps_norm - 1.0) <= 1e-10 and state_error <= 1e-9
        metadata = {
            "qubits": args.qubits, "layers": args.layers, "seed": args.seed,
            "topology": args.topology, "gates": len(circuit.queue), "depth": circuit.depth,
            "qasm_sha256": qasm_hash, "threads": args.threads,
            "svd_cutoff": 0.0, "max_bond": None, "max_bond_observed": mps.max_bond(),
            "precision": str(mps_vector.dtype), "reference": "independent Qibo NumPy dense state (correctness only)",
            "fidelity_vs_exact": fidelity, "state_norm_squared": mps_norm,
            "max_state_error_after_global_phase_alignment": state_error,
            "state_correctness_pass": bool(state_pass), "threadpools": threadpool_info(),
            "packages": {p: importlib.metadata.version(p) for p in ("qibo", "qibotn", "quimb", "numpy", "scipy")},
            "source_sha256": hashlib.sha256((ROOT / "src/qibotn/backends/quimb.py").read_bytes()).hexdigest(),
            "auxiliary_source_sha256": hashlib.sha256((ROOT / "src/qibotn/backends/_exact_numpy.py").read_bytes()).hexdigest(),
            "linear_compiler_source_sha256": hashlib.sha256((ROOT / "src/qibotn/backends/_linear_cnot.py").read_bytes()).hexdigest(),
            "structured_kernel_sha256": hashlib.sha256((ROOT / "src/qibotn/backends/_structured_numpy.py").read_bytes()).hexdigest(),
            "cnot_schedule_sha256": hashlib.sha256((ROOT / "src/qibotn/backends/_cnot_schedule.py").read_bytes()).hexdigest(),
            "cpu_affinity": process.cpu_affinity(), "requested_affinity_cpu": args.affinity_cpu,
            "requested_affinity_cpus": args.affinity_cpus,
            "mps_execution": backend.mps_execution, "state_preparation": preparation,
            "timing_scope": "QiboTN conversion + full gate application + normalized observable evaluation; excludes imports and independent correctness reference",
            "scope": "local multi-topology validation, not formal 30-qubit acceptance",
            "optimized_preparation_full_state_checked": True,
        }
        metadata_path = args.output / "metadata.json"
        if args.resume and metadata_path.exists():
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity_fields = ("qubits", "layers", "seed", "topology", "threads", "qasm_sha256", "svd_cutoff", "max_bond", "source_sha256", "auxiliary_source_sha256", "linear_compiler_source_sha256", "structured_kernel_sha256", "cnot_schedule_sha256", "cpu_affinity")
            if any(original[key] != metadata[key] for key in identity_fields):
                raise RuntimeError("Resume input/config/source identity mismatch")
            write_json(args.output / "metadata_resume.json", metadata)
        else:
            write_json(metadata_path, metadata)
        if not state_pass:
            raise RuntimeError("Independent state reference check failed")
        if mps_vector.dtype != np.dtype("complex128"):
            raise RuntimeError("Benchmark requires complex128 on both paths")
        del reference_state, mps_vector, aligned, mps
        rows_path = args.output / "rows.json"
        rows = json.loads(rows_path.read_text(encoding="utf-8")) if args.resume and rows_path.exists() else []
        identities = [(row["family"], row["repeat"], row["method"]) for row in rows]
        if len(set(identities)) != len(identities) or any(not row["correctness_pass"] for row in rows):
            raise RuntimeError("Existing evidence contains duplicates or a correctness failure; cannot resume")
        summaries = {}
        # Restore the dense reference only for correctness, outside all timers.
        reference_state = NumpyBackend().execute_circuit(circuit).state()
        for family in ("zz4", "mixed"):
            labels, sites, coefficients = observable(args.qubits, family)
            exact_value = dense_expectation(reference_state, args.qubits, labels, sites, coefficients)
            timings = {method: [row["runtime_s"] for row in rows
                                if row["family"] == family and row["repeat"] > 0 and row["method"] == method]
                       for method in ("reference", "shared_environments")}
            for repeat in range(args.repeats + 1):
                # Balanced order; repeat zero is an explicit untimed warm-up.
                methods = ["reference", "shared_environments"]
                if repeat % 2:
                    methods.reverse()
                for method in methods:
                    if (family, repeat, method) in identities:
                        continue
                    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0.0, expectation_method=method)
                    started = time.perf_counter()
                    value = backend.exp_value_observable_symbolic(circuit, labels, sites, coefficients, args.qubits)
                    elapsed = time.perf_counter() - started
                    error = abs(value - exact_value)
                    row = {"family": family, "terms": len(labels), "repeat": repeat, "method": method,
                           "runtime_s": elapsed, "value": value, "exact_value": exact_value,
                           "abs_error": error, "correctness_pass": error <= 1e-10,
                           "state_norm_squared": backend.last_expectation_diagnostics["state_norm_squared"],
                           "state_preparation": backend.last_expectation_diagnostics["state_preparation"],
                           "rss_mib_at_end": psutil.Process().memory_info().rss / 2**20}
                    rows.append(row)
                    write_json(args.output / "rows.json", rows)
                    print(json.dumps({"topology": args.topology, "qubits": args.qubits, **row}), flush=True)
                    if error > 1e-10:
                        raise RuntimeError("Observable reference check failed")
                    if repeat:
                        timings[method].append(elapsed)
            baseline = statistics.mean(timings["reference"])
            optimized = statistics.mean(timings["shared_environments"])
            summaries[family] = {"reference_mean_s": baseline, "optimized_mean_s": optimized,
                                 "code_speedup_same_threads": baseline / optimized,
                                 "reference_stdev_s": statistics.stdev(timings["reference"]),
                                 "optimized_stdev_s": statistics.stdev(timings["shared_environments"]),
                                 "all_correct": True, "repeats": args.repeats, "terms": len(labels)}
        write_json(args.output / "summary.json", {"metadata": metadata, "families": summaries,
                                                  "resumed_missing_measurements_only": bool(args.resume)})


def suite(args):
    import psutil
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists; choose a new directory to preserve raw evidence")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = ROOT / "src/qibotn/backends/quimb.py"
    snapshot = args.output / "source_snapshot.py"
    auxiliary = ROOT / "src/qibotn/backends/_exact_numpy.py"
    auxiliary_snapshot = args.output / "exact_numpy_snapshot.py"
    linear = ROOT / "src/qibotn/backends/_linear_cnot.py"
    linear_snapshot = args.output / "linear_cnot_snapshot.py"
    extras = [(ROOT / "src/qibotn/backends" / name,
               args.output / (name.lstrip("_").removesuffix(".py") + "_snapshot.py"))
              for name in ("_structured_numpy.py", "_cnot_schedule.py")]
    if args.resume:
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(snapshot.read_bytes()).digest():
            raise RuntimeError("Resume source differs from its preserved snapshot")
        if hashlib.sha256(auxiliary.read_bytes()).digest() != hashlib.sha256(auxiliary_snapshot.read_bytes()).digest():
            raise RuntimeError("Resume auxiliary source differs from its preserved snapshot")
        if hashlib.sha256(linear.read_bytes()).digest() != hashlib.sha256(linear_snapshot.read_bytes()).digest():
            raise RuntimeError("Resume linear compiler differs from its preserved snapshot")
        for extra, archived in extras:
            if hashlib.sha256(extra.read_bytes()).digest() != hashlib.sha256(archived.read_bytes()).digest():
                raise RuntimeError("Resume extra optimization source differs from its snapshot")
    else:
        # Archive rather than rewrite the exact source used by these jobs.
        shutil.copyfile(source, snapshot)
        shutil.copyfile(auxiliary, auxiliary_snapshot)
        shutil.copyfile(linear, linear_snapshot)
        for extra, archived in extras:
            shutil.copyfile(extra, archived)
        shutil.copyfile(Path(__file__), args.output / "evaluate_snapshot.py")
    prior_entries = []
    if args.resume:
        prior_summary = json.loads((args.output / "suite_summary.json").read_text(encoding="utf-8"))
        prior_entries = prior_summary["entries"]
        backup = args.output / "suite_summary_before_resume.json"
        if backup.exists():
            raise ValueError("This output already used its single allowed resume")
        write_json(backup, prior_summary)
    all_summaries = []
    started = time.monotonic()
    stop = False
    for n in args.sizes:
        for topology in args.topologies:
            for threads in args.thread_counts:
                if stop:
                    break
                prior = next((entry for entry in prior_entries
                              if (entry["qubits"], entry["topology"], entry["threads"]) == (n, topology, threads)), None)
                if prior and prior["exit_code"] == 0:
                    original = prior["result"]["metadata"]
                    if original["seed"] != args.seed or original["layers"] != args.layers:
                        raise RuntimeError("Completed resume entries do not match the requested workload")
                    all_summaries.append(prior)
                    continue
                if prior and prior["budget_stop_reason"] != "case_time_budget_exceeded":
                    raise RuntimeError("Only a known per-case timeout can be resumed, not a correctness failure")
                folder = args.output / f"q{n}_{topology}_t{threads}"
                folder.mkdir(exist_ok=args.resume)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT / "src")
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "QUIMB_NUM_PROCS"):
                    env[name] = str(threads)
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(folder),
                           "--qubits", str(n), "--layers", str(args.layers), "--topology", topology,
                           "--seed", str(args.seed), "--threads", str(threads), "--repeats", str(args.repeats)]
                if args.resume:
                    command.append("--resume")
                if args.affinity_cpu is not None:
                    command.extend(["--affinity-cpu", str(args.affinity_cpu)])
                elif args.affinity_cpus is not None:
                    command.extend(["--affinity-cpus", *[str(k) for k in args.affinity_cpus]])
                log = (folder / ("run_resume.log" if args.resume else "run.log")).open("w", encoding="utf-8")
                print(f"START q={n} topology={topology} threads={threads}", flush=True)
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                job_started = time.monotonic()
                peak_rss = 0
                reason = None
                while process.poll() is None:
                    try:
                        tracked = psutil.Process(process.pid)
                        rss = tracked.memory_info().rss + sum(p.memory_info().rss for p in tracked.children(recursive=True))
                        peak_rss = max(peak_rss, rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    if peak_rss > args.memory_mib * 2**20:
                        reason = "memory_budget_exceeded"
                    elif time.monotonic() - job_started > args.case_seconds:
                        reason = "case_time_budget_exceeded"
                    elif time.monotonic() - started > args.total_seconds:
                        reason = "suite_time_budget_exceeded"
                        stop = True
                    if reason:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            child.kill()
                        process.kill()
                        break
                    time.sleep(0.2)
                code = process.wait()
                log.close()
                entry = {"qubits": n, "topology": topology, "threads": threads,
                         "exit_code": code, "peak_rss_mib": peak_rss / 2**20,
                         "budget_stop_reason": reason}
                if prior:
                    entry["prior_attempt"] = {key: prior[key] for key in ("exit_code", "peak_rss_mib", "budget_stop_reason")}
                if (folder / "summary.json").exists() and code == 0:
                    entry["result"] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                all_summaries.append(entry)
                write_json(args.output / "suite_checkpoint.json", all_summaries)
                print(f"DONE q={n} topology={topology} threads={threads} exit={code} stop={reason}", flush=True)
            if stop:
                break
        if stop:
            break
    report = {"scope": "local validation only; large-workload formal acceptance remains pending",
              "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "auxiliary_source_sha256": hashlib.sha256(auxiliary_snapshot.read_bytes()).hexdigest(),
              "linear_compiler_source_sha256": hashlib.sha256(linear_snapshot.read_bytes()).hexdigest(),
              "structured_kernel_sha256": hashlib.sha256(extras[0][1].read_bytes()).hexdigest(),
              "cnot_schedule_sha256": hashlib.sha256(extras[1][1].read_bytes()).hexdigest(),
              "requested_affinity_cpu": args.affinity_cpu,
              "requested_affinity_cpus": args.affinity_cpus,
              "measurement_repeats": args.repeats,
              "budgets": {"memory_mib": args.memory_mib, "case_seconds": args.case_seconds,
                          "total_seconds": args.total_seconds},
              "no_historical_thread_speedup_reused": True, "no_cutoff_relaxation": True,
              "elapsed_wall_s": time.monotonic() - started, "entries": all_summaries,
              "requested_topologies": args.topologies,
              "all_requested_jobs_attempted": len(all_summaries) == len(args.sizes) * len(args.topologies) * len(args.thread_counts),
              "all_requested_jobs_completed": (len(all_summaries) == len(args.sizes) * len(args.topologies) * len(args.thread_counts)
                                               and all(e["exit_code"] == 0 for e in all_summaries)),
              "all_completed_jobs_pass": all(e["exit_code"] == 0 for e in all_summaries)}
    write_json(args.output / "suite_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true", help="one audited continuation after a per-case timeout; no replacement of measurements")
    parser.add_argument("--sizes", type=int, nargs="+", default=[12])
    parser.add_argument("--topologies", nargs="+", choices=["chain", "ring", "long_range", "all_to_all"],
                        default=["chain", "ring", "long_range", "all_to_all"])
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--topology", default="chain")
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity-cpu", type=int, default=None,
                        help="Pin this worker and both timing paths to one available logical CPU")
    parser.add_argument("--affinity-cpus", type=int, nargs="+", default=None,
                        help="Identical logical CPU set for both paths, including thread tuning")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-mib", type=int, default=1000)
    parser.add_argument("--case-seconds", type=int, default=120)
    parser.add_argument("--total-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.affinity_cpu is not None and args.affinity_cpus is not None:
        parser.error("Choose either one logical CPU or a logical CPU set")
    if args.repeats < 2 or any(n < 3 or n > 20 for n in args.sizes) or not 3 <= args.qubits <= 20:
        parser.error("local dense verification requires 3..20 qubits and at least two measured repeats")
    worker(args) if args.worker else suite(args)


if __name__ == "__main__":
    main()
