"""Observe an existing 30q/9998-gate workload under strict local limits.

This is an instrumented resource diagnosis, NOT a publishable speed test.
It never lowers precision, truncates a state, retries or chooses easy outputs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def worker(args):
    import numpy as np
    import psutil
    import quimb.tensor as qtn
    from threadpoolctl import threadpool_limits, threadpool_info
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit, make_observable
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from qibotn.backends.quimb import QuimbNumpyBackend
    psutil.Process().cpu_affinity([2, 4, 6, 8])
    circuit = build_circuit(Workload(30, 112))
    qasm = circuit.to_qasm()
    (args.output / "input.qasm").write_text(qasm, encoding="utf-8")
    info = {
        "status": "preparing", "instrumented_resource_probe_not_formal_timing": True,
        "qubits": 30, "gates": len(circuit.queue), "depth": circuit.depth,
        "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "method": args.method, "precision": "complex128", "cutoff": 0,
        "max_bond": None, "threads": 1, "affinity": [2, 4, 6, 8],
        "original_workload": "existing asc_large_cpu Workload(30,112), not a new specially easy circuit",
    }
    save(args.output / "probe.json", info)
    if args.method == "upstream_typo_only":
        klass, metadata = load_reference(args.output)
        backend = klass()
        info["reference"] = metadata
    else:
        backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    original_apply = qtn.CircuitMPS.apply_gate
    progress = {"completed_quimb_apply_gate_calls": 0, "observed_max_bond": 1,
                "latest_mps_array_bytes": 0}
    last_write = 0.
    last_state = None
    started = time.perf_counter()
    def wrapped(self, *gate_args, **gate_kwargs):
        nonlocal last_write, last_state
        now = time.perf_counter()
        if now - last_write >= .5:
            progress["elapsed_instrumented_s"] = now - started
            progress["in_flight_gate"] = str(gate_args[0]) if gate_args else "unknown"
            save(args.output / "progress.json", progress)
            last_write = now
        value = original_apply(self, *gate_args, **gate_kwargs)
        progress["completed_quimb_apply_gate_calls"] += 1
        last_state = self._psi
        progress["observed_max_bond"] = max(progress["observed_max_bond"], last_state.max_bond())
        progress["latest_mps_array_bytes"] = sum(np.asarray(t.data).nbytes for t in last_state.tensor_map.values())
        return value
    qtn.CircuitMPS.apply_gate = wrapped
    with threadpool_limits(limits=1):
        info["threadpools"] = threadpool_info()
        info["status"] = "running"
        save(args.output / "probe.json", info)
        labels, sites, coefficients = make_observable(30)
        value = float(backend.exp_value_observable_symbolic(circuit, labels, sites, coefficients, 30))
        info["instrumented_elapsed_s"] = time.perf_counter() - started
        if last_state is None:
            raise RuntimeError("No native MPS state was observed")
        norm = float(np.real(last_state.H @ last_state))
        info.update({"status": "completed_probe", "observable": value, "norm_squared": norm,
                     "progress": progress, "full_independent_30q_correctness_proven": False})
        save(args.output / "probe.json", info)
        save(args.output / "progress.json", progress)
        print(json.dumps(info), flush=True)
    qtn.CircuitMPS.apply_gate = original_apply


def suite(args):
    import psutil
    if args.output.exists():
        raise RuntimeError("Fresh directory required")
    args.output.mkdir(parents=True)
    paths = [Path(__file__), HERE / "upstream_reference.py",
             ROOT / "benchmarks/asc_large_cpu/benchmark.py"]
    paths += [ROOT / "src/qibotn/backends" / name for name in
              ("quimb.py", "_exact_numpy.py", "_linear_cnot.py", "_structured_numpy.py", "_cnot_schedule.py")]
    for path in paths:
        shutil.copyfile(path, args.output / (path.stem + "_snapshot.py"))
    report = {"recorded_at": datetime.now().astimezone().isoformat(),
              "instrumented_resource_probe_not_formal_timing": True,
              "formal_30q_acceptance_complete": False, "publishable_speedup": None,
              "policy": "one attempt per path, no truncation, no dense 30q allocation, bounded diagnosis",
              "budgets_per_path": {"seconds": args.seconds, "memory_mib": args.memory_mib,
                                   "minimum_system_available_mib": 800},
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "host": {"total_memory_bytes": psutil.virtual_memory().total,
                       "available_memory_bytes_at_start": psutil.virtual_memory().available}, "paths": []}
    save(args.output / "summary.json", report)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "QUIMB_NUM_PROCS"):
        env[key] = "1"
    for method in ("upstream_typo_only", "frozen_optimized"):
        destination = args.output / method
        destination.mkdir()
        if psutil.virtual_memory().available < 1200 * 2**20:
            report["paths"].append({"method": method, "status": "not_started_low_available_memory"})
            save(args.output / "summary.json", report)
            continue
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker",
                   "--output", str(destination), "--method", method]
        started, peak, reason = time.monotonic(), 0, None
        with (destination / "run.log").open("w", encoding="utf-8") as log:
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
                        reason = "resource_diagnosis_time_budget_exceeded"
                    if reason:
                        for child in tree.children(recursive=True):
                            child.kill()
                        process.kill()
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(.05)
            code = process.wait()
        record = {"method": method, "status": "budget_stopped" if reason else "exited",
                  "exit_code": code, "stop_reason": reason, "elapsed_s": time.monotonic() - started,
                  "peak_process_tree_rss_mib": peak / 2**20}
        for name in ("progress", "probe"):
            path = destination / (name + ".json")
            if path.exists():
                record[name] = json.loads(path.read_text(encoding="utf-8"))
        report["paths"].append(record)
        save(args.output / "summary.json", report)
        print(json.dumps(record), flush=True)
    report["status"] = "diagnosis_complete_not_acceptance"
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["upstream_typo_only", "frozen_optimized"])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--memory-mib", type=float, default=900)
    arguments = parser.parse_args()
    if arguments.worker and arguments.method is None:
        parser.error("Worker requires a method")
    worker(arguments) if arguments.worker else suite(arguments)
