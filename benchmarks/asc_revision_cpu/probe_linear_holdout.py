"""Unseen-seed/random-connectivity checks, including non-CNOT entanglers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import psutil
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from threadpoolctl import threadpool_info, threadpool_limits

from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import ROOT, dense_expectation, observable


def build(kind, seed, n=12, layers=6):
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            c.add([gates.RY(q, float(rng.uniform(-np.pi, np.pi))),
                   gates.RZ(q, float(rng.uniform(-np.pi, np.pi)))])
        for _ in range(n*(n-1)//2):
            pair = [int(k) for k in rng.choice(n, 2, replace=False)]
            gate_type = gates.CNOT if kind == "random_cnot" else (gates.CNOT, gates.CY, gates.CZ)[int(rng.integers(3))]
            c.add(gate_type(*pair))
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[801271,991573])
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve prior evidence")
    args.output.mkdir(parents=True)
    psutil.Process().cpu_affinity([2,4,6,8])
    result = {"scope":"additional local holdout, not 30q acceptance", "status":"running", "jobs":[], "rows":[], "hashes":{}}
    def save():
        (args.output / "holdout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for path in (Path(__file__), *[ROOT / "src/qibotn/backends" / name for name in
            ("quimb.py", "_exact_numpy.py", "_linear_cnot.py", "_structured_numpy.py", "_cnot_schedule.py")]):
        result["hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path,args.output / (path.stem+"_snapshot.py"))
    started = time.monotonic()
    with threadpool_limits(limits=1):
        result["threadpools"]=threadpool_info()
        result["affinity"]=psutil.Process().cpu_affinity()
        b=QuimbNumpyBackend()
        for kind in ("random_cnot","mixed_entanglers"):
            for seed in args.seeds:
                c=build(kind,seed)
                qasm=c.to_qasm()
                reference=NumpyBackend().execute_circuit(c).state()
                b.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
                state,prep=b._prepare_optimized_mps(c)
                actual=np.asarray(state.to_dense()).reshape(-1)
                norm=float(np.vdot(actual,actual).real)
                phase=np.vdot(reference,actual)
                fidelity=float(abs(phase)**2 / (norm*np.vdot(reference,reference).real))
                error=float(np.max(np.abs(actual*np.exp(-1j*np.angle(phase))-reference)))
                job={"kind":kind,"seed":seed,"qubits":12,"gates":len(c.queue),"depth":c.depth,
                     "qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),"norm_squared":norm,
                     "fidelity":fidelity,"max_state_error":error,"preparation":prep}
                result["jobs"].append(job)
                save()
                if abs(norm-1)>1e-10 or fidelity<1-1e-10 or error>1e-9:
                    result["status"]="state_failure"
                    save()
                    raise RuntimeError("Holdout full-state mismatch")
                labels,sites,coeff=observable(12,"zz4")
                exact=dense_expectation(reference,12,labels,sites,coeff)
                samples={m:[] for m in ("reference","shared_environments")}
                for repeat in range(4):
                    methods=list(samples)
                    if repeat%2:
                        methods.reverse()
                    for method in methods:
                        if time.monotonic()-started>180 or psutil.Process().memory_info().rss>900*2**20:
                            result["status"]="budget_stop"
                            save()
                            raise RuntimeError("Holdout resource budget exceeded")
                        b.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0,expectation_method=method)
                        begin=time.perf_counter()
                        value=b.exp_value_observable_symbolic(c,labels,sites,coeff,12)
                        elapsed=time.perf_counter()-begin
                        error=abs(value-exact)
                        result["rows"].append({"kind":kind,"seed":seed,"repeat":repeat,"method":method,
                                               "runtime_s":elapsed,"value":value,"exact_value":exact,"abs_error":error,
                                               "diagnostics":b.last_expectation_diagnostics})
                        save()
                        if error>1e-10:
                            result["status"]="observable_failure"
                            save()
                            raise RuntimeError("Holdout observable mismatch")
                        if repeat:
                            samples[method].append(elapsed)
                assert c.to_qasm()==qasm
                means={m:statistics.mean(t) for m,t in samples.items()}
                job["means_s"]=means
                job["stdev_s"]={m:statistics.stdev(t) for m,t in samples.items()}
                job["speedup"]=means["reference"]/means["shared_environments"]
                print(json.dumps(job),flush=True)
        result["status"]="pass"
        result["elapsed_s"]=time.monotonic()-started
        save()


if __name__=="__main__":
    main()
