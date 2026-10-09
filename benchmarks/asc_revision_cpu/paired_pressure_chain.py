"""Exact CPU pressure benchmark on the existing deep hardware-efficient input.

Calibration is explicitly not acceptance; production timing includes all gates
and the original four ZZ terms, but not imports or independent dense checks.
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
CORE = [ROOT / 'src/qibotn/backends' / name for name in
        ('quimb.py', '_exact_numpy.py', '_linear_cnot.py', '_structured_numpy.py', '_cnot_schedule.py', '_mixed_schedule.py')]
HELPERS = [Path(__file__), HERE / 'upstream_reference.py', HERE / 'evaluate.py',
           ROOT / 'benchmarks/asc_large_cpu/benchmark.py']


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def check_state(actual, oracle, np):
    norm = float(np.vdot(actual, actual).real)
    overlap = np.vdot(oracle, actual)
    fidelity = float(abs(overlap)**2 / (norm * np.vdot(oracle, oracle).real))
    error = float(np.max(np.abs(actual * np.exp(-1j * np.angle(overlap)) - oracle)))
    if (not all(map(math.isfinite, (norm, fidelity, error))) or abs(norm-1) > 1e-10 or
            abs(fidelity-1) > 1e-10 or error > 1e-9):
        raise RuntimeError('Independent full-state correctness failed')
    return {'norm_squared': norm, 'fidelity': fidelity, 'max_state_error': error}


def worker(args):
    import numpy as np
    import psutil
    from qibo.backends import NumpyBackend
    from qibotn.backends.quimb import QuimbNumpyBackend
    from threadpoolctl import threadpool_info, threadpool_limits
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_revision_cpu.evaluate import dense_expectation
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit, make_observable
    psutil.Process().cpu_affinity(args.affinity)
    circuit = build_circuit(Workload(args.qubits, args.layers))
    qasm = circuit.to_qasm()
    (args.output / 'input.qasm').write_text(qasm, encoding='utf-8')
    reference_class, reference_metadata = load_reference(args.output)
    methods = {'upstream_typo_only': reference_class(), 'integrated': QuimbNumpyBackend()}
    for backend in methods.values():
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    labels, sites, coefficients = make_observable(args.qubits)
    data = {'status': 'preparing', 'calibration_only': args.calibration,
        'scope': 'hardware-limited 20q deep pressure acceptance, not a 30q speed claim',
        'workload': 'existing asc_large_cpu hardware-efficient RX/RY plus entangling brickwork',
        'qubits': args.qubits, 'layers': args.layers, 'gates': len(circuit.queue), 'depth': circuit.depth,
        'qasm_sha256': hashlib.sha256(qasm.encode()).hexdigest(), 'precision': 'complex128',
        'cutoff': 0, 'max_bond': None, 'threads': 1, 'affinity': args.affinity,
        'warmup_repeats': 0 if args.calibration else 1,
        'measured_repeats': 0 if args.calibration else args.repeats,
        'reference': reference_metadata, 'rows': [], 'full_state': {},
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (*CORE, *HELPERS)},
        'timing_scope': 'all circuit conversion/compiler, state preparation and original four ZZ terms; excludes imports and independent oracle',
        'gate_counts': {name: sum(type(g).__name__ == name for g in circuit.queue)
                        for name in sorted({type(g).__name__ for g in circuit.queue})}}
    path = args.output / 'result.json'
    save(path, data)
    with threadpool_limits(limits=1):
        data['threadpools'] = threadpool_info()
        exact = None
        if not args.calibration:
            data['status'] = 'independent_dense_oracle'
            save(path, data)
            oracle = NumpyBackend().execute_circuit(circuit).state()
            exact = dense_expectation(oracle, args.qubits, labels, sites, coefficients)
            for method, backend in methods.items():
                data['status'] = 'full_state_' + method
                save(path, data)
                if method == 'integrated':
                    state, preparation = backend._prepare_optimized_mps(circuit)
                else:
                    state = backend._qibo_circuit_to_quimb(circuit, quimb_circuit_type=backend.circuit_ansatz,
                        gate_opts={'max_bond': None, 'cutoff': 0}).psi
                    preparation = {'engine': 'official_native'}
                actual = np.asarray(state.to_dense()).reshape(-1)
                data['full_state'][method] = {**check_state(actual, oracle, np),
                    'max_bond_observed': state.max_bond(), 'preparation': preparation}
                save(path, data)
                del state, actual
            del oracle
        repeats = range(1) if args.calibration else range(args.repeats + 1)
        for repeat in repeats:
            order = list(methods)
            if repeat % 2:
                order.reverse()
            for method in order:
                data['status'] = 'timing_' + method
                save(path, data)
                begin = time.perf_counter()
                value = float(methods[method].exp_value_observable_symbolic(
                    circuit, labels, sites, coefficients, args.qubits))
                elapsed = time.perf_counter() - begin
                error = None if exact is None else abs(value - exact)
                row = {'method': method, 'repeat': repeat, 'runtime_s': elapsed,
                    'value': value, 'exact_value': exact, 'abs_error': error}
                if method == 'integrated':
                    row['diagnostics'] = methods[method].last_expectation_diagnostics
                data['rows'].append(row)
                save(path, data)
                print(json.dumps({k: row[k] for k in ('method', 'repeat', 'runtime_s', 'abs_error')}), flush=True)
                if not math.isfinite(value) or elapsed <= 0 or (error is not None and error > 1e-10):
                    raise RuntimeError('Observable correctness failed')
        if circuit.to_qasm() != qasm:
            raise RuntimeError('Original input changed')
        if args.calibration:
            data['status'] = 'calibration_complete_not_acceptance'
            data['publishable_speedup'] = None
        else:
            means = {method: statistics.mean(r['runtime_s'] for r in data['rows']
                if r['method'] == method and r['repeat'] > 0) for method in methods}
            data['means_s'] = means
            data['stdev_s'] = {method: statistics.stdev(r['runtime_s'] for r in data['rows']
                if r['method'] == method and r['repeat'] > 0) for method in methods}
            data['speedup'] = means['upstream_typo_only'] / means['integrated']
            data['status'] = 'pass'
        save(path, data)


def suite(args):
    import psutil
    if args.output.exists():
        raise RuntimeError('Fresh directory required; preserve prior evidence')
    if psutil.virtual_memory().available < (args.memory_mib + 1000) * 2**20:
        raise RuntimeError('Not started: requested process allowance plus 1000 MiB safety headroom unavailable')
    args.output.mkdir(parents=True)
    save(args.output / 'run_metadata.json', {'started_at': datetime.now().astimezone().isoformat(),
        'controller_pid': os.getpid(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'host_total_memory_bytes': psutil.virtual_memory().total,
        'host_available_memory_bytes': psutil.virtual_memory().available,
        'policy': 'fixed predeclared input, same CPU and precision, no candidate retry or output selection'})
    for path in (*CORE, *HELPERS):
        shutil.copyfile(path, args.output / (path.stem + '_snapshot.py'))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'QUIMB_NUM_PROCS'):
        env[key] = '1'
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', '--output', str(args.output),
        '--qubits', str(args.qubits), '--layers', str(args.layers), '--repeats', str(args.repeats),
        '--affinity', *map(str, args.affinity)]
    if args.calibration:
        command.append('--calibration')
    begin, peak, reason = time.monotonic(), 0, None
    with (args.output / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                tree = psutil.Process(process.pid)
                peak = max(peak, tree.memory_info().rss + sum(c.memory_info().rss for c in tree.children(recursive=True)))
                if peak > args.memory_mib * 2**20:
                    reason = 'process_tree_memory_budget_exceeded'
                elif psutil.virtual_memory().available < 1000 * 2**20:
                    reason = 'system_available_memory_below_1000_MiB'
                elif time.monotonic() - begin > args.seconds:
                    reason = 'wall_clock_budget_exceeded'
                if reason:
                    for child in tree.children(recursive=True):
                        child.kill()
                    process.kill()
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.1)
        code = process.wait()
    output = args.output / 'result.json'
    expected = 'calibration_complete_not_acceptance' if args.calibration else 'pass'
    passed = code == 0 and reason is None and output.exists() and json.loads(output.read_text(encoding='utf-8'))['status'] == expected
    report = {'status': 'pass' if passed else 'incomplete_or_failed', 'exit_code': code, 'stop_reason': reason,
        'elapsed_s': time.monotonic() - begin, 'peak_process_tree_rss_mib': peak / 2**20,
        'calibration_only': args.calibration, 'formal_30q_acceptance_complete': False,
        'budgets': {'seconds': args.seconds, 'memory_mib': args.memory_mib}}
    save(args.output / 'summary.json', report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qubits', type=int, default=20)
    parser.add_argument('--layers', type=int, default=169)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--affinity', type=int, nargs='+', default=[2,4,6,8])
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('--memory-mib', type=float, default=1800)
    args = parser.parse_args()
    if not 2 <= args.qubits <= 20 or args.layers < 1 or args.repeats < 3:
        parser.error('Need 2..20 qubits, >=1 layer and >=3 measured repetitions')
    worker(args) if args.worker else suite(args)
