"""Resume missing calls, preserving the stopped segment and its raw evidence.

No circuit, production source, scalar reference or accepted row is replaced.
Each backend gets a new unmeasured warmup after the process restart. All
resumed measured calls still construct a new state, exactly as before.
"""
import argparse
from copy import deepcopy
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

from benchmarks.asc_revision_cpu.paired_pressure_chain import ROOT, save

METHODS = ('upstream_typo_only', 'integrated')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(repeats=3):
    return [(method, repeat) for repeat in range(repeats + 1)
            for method in (METHODS[::-1] if repeat % 2 else METHODS)]


def validate_checkpoint(data, summary):
    if (summary.get('stop_reason') != 'wall_clock_budget_exceeded' or
            summary.get('status') != 'incomplete_or_failed'):
        raise ValueError('Only an explicitly time-stopped checkpoint is resumable')
    if (data.get('qubits'), data.get('layers'), data.get('gates'), data.get('depth')) != (20, 169, 9991, 677):
        raise ValueError('Unexpected pressure input')
    if (data.get('precision') != 'complex128' or data.get('cutoff') != 0 or
            data.get('max_bond') is not None or data.get('threads') != 1 or
            data.get('calibration_only') is not False or data.get('measured_repeats') != 3):
        raise ValueError('Unexpected numerical or repetition configuration')
    if set(data.get('full_state', {})) != set(METHODS):
        raise ValueError('Both independent full-state checks must be durable')
    for state in data['full_state'].values():
        if (not all(math.isfinite(state[k]) for k in ('norm_squared', 'fidelity', 'max_state_error')) or
                abs(state['norm_squared'] - 1) > 1e-10 or abs(state['fidelity'] - 1) > 1e-10 or
                state['max_state_error'] > 1e-9):
            raise ValueError('Checkpoint full-state correctness is not acceptable')
    rows = data['rows']
    keys = [(r['method'], r['repeat']) for r in rows]
    if not rows or keys != plan()[:len(rows)] or len(keys) >= len(plan()):
        raise ValueError('Checkpoint is not a nonempty strict prefix of the declared plan')
    exact = rows[0]['exact_value']
    if not isinstance(exact, (int, float)) or not math.isfinite(exact):
        raise ValueError('Missing independent scalar reference')
    for row in rows:
        if (row['exact_value'] != exact or not all(math.isfinite(row[k]) for k in
                ('runtime_s', 'value', 'abs_error')) or row['runtime_s'] <= 0 or
                abs(row['value'] - exact) > 1e-10 or row['abs_error'] > 1e-10):
            raise ValueError('Invalid completed row')
    return plan()[len(rows):], exact


def worker(args):
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit, make_observable
    path = args.output / 'result.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    source = json.loads((args.output / 'prior_segment/result.json').read_text(encoding='utf-8'))
    summary = json.loads((args.output / 'prior_segment/summary.json').read_text(encoding='utf-8'))
    remaining, exact = validate_checkpoint(source, summary)
    for relative, expected in source['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Measured source changed before restart: ' + relative)
    psutil.Process().cpu_affinity(data['affinity'])
    circuit = build_circuit(Workload(data['qubits'], data['layers']))
    qasm = circuit.to_qasm()
    if hashlib.sha256(qasm.encode()).hexdigest() != data['qasm_sha256']:
        raise ValueError('Regenerated input changed')
    reference, metadata = load_reference(args.output)
    if metadata != data['reference']:
        raise ValueError('Pinned official reference changed')
    methods = {'upstream_typo_only': reference(), 'integrated': QuimbNumpyBackend()}
    for backend in methods.values():
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    labels, sites, coefficients = make_observable(data['qubits'])

    def call(method, repeat, warmup=False):
        data['status'] = ('restart_warmup_' if warmup else 'timing_') + method
        save(path, data)
        begin = time.perf_counter()
        value = float(methods[method].exp_value_observable_symbolic(
            circuit, labels, sites, coefficients, data['qubits']))
        elapsed = time.perf_counter() - begin
        row = {'method': method, 'repeat': repeat, 'runtime_s': elapsed, 'value': value,
               'exact_value': exact, 'abs_error': abs(value-exact), 'segment': 'resumed_process'}
        if method == 'integrated':
            row['diagnostics'] = methods[method].last_expectation_diagnostics
        if not math.isfinite(value) or elapsed <= 0 or row['abs_error'] > 1e-10:
            save(args.output / 'failed_call.json', row)
            raise RuntimeError('Resumed observable correctness failed')
        (data['restart_warmups'] if warmup else data['rows']).append(row)
        save(path, data)
        print(json.dumps({key: row[key] for key in ('method', 'repeat', 'runtime_s', 'abs_error', 'segment')}), flush=True)

    with threadpool_limits(limits=1):
        data['resumed_threadpools'] = threadpool_info()
        if any(pool['num_threads'] != 1 for pool in data['resumed_threadpools']):
            raise RuntimeError('Resumed BLAS thread configuration mismatch')
        # Explicit fresh-process warmups; never count them as measured rows.
        for method in METHODS:
            call(method, -1, warmup=True)
        for method, repeat in remaining:
            call(method, repeat)
    if circuit.to_qasm() != qasm:
        raise RuntimeError('Input changed during timing')
    for relative, expected in source['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Measured source changed during restart: ' + relative)
    means = {method: statistics.mean(r['runtime_s'] for r in data['rows']
             if r['method'] == method and r['repeat'] > 0) for method in METHODS}
    data.update(status='pass', means_s=means,
        stdev_s={method: statistics.stdev(r['runtime_s'] for r in data['rows']
                 if r['method'] == method and r['repeat'] > 0) for method in METHODS},
        speedup=means['upstream_typo_only']/means['integrated'])
    save(path, data)


def suite(args):
    import psutil
    if args.output.exists():
        raise FileExistsError('Fresh continuation directory required')
    data = json.loads((args.source / 'result.json').read_text(encoding='utf-8'))
    summary = json.loads((args.source / 'summary.json').read_text(encoding='utf-8'))
    remaining, _ = validate_checkpoint(data, summary)
    if psutil.virtual_memory().available < (args.memory_mib + 1000) * 2**20:
        raise RuntimeError('Insufficient safety headroom; worker not started')
    for relative, expected in data['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Current measured source mismatch: ' + relative)
    args.output.mkdir(parents=True)
    shutil.copytree(args.source, args.output / 'prior_segment')
    for source in args.source.iterdir():
        if source.is_file() and (source.name.endswith('_snapshot.py') or source.name in
                ('input.qasm', 'upstream_quimb_original.py', 'upstream_quimb_typo_only.py')):
            shutil.copyfile(source, args.output / source.name)
    shutil.copyfile(Path(__file__), args.output / 'resume_pressure_chain_snapshot.py')
    data = deepcopy(data)
    data['continuation'] = {'original_directory': str(args.source.resolve()),
        'original_result_sha256': digest(args.source / 'result.json'),
        'original_summary_sha256': digest(args.source / 'summary.json'),
        'resume_script_sha256': digest(Path(__file__)),
        'preserved_row_count': len(data['rows']), 'remaining_plan': remaining,
        'original_stop_reason': summary['stop_reason'], 'original_exit_code': summary['exit_code'],
        'same_production_bytes_and_input': True, 'fresh_process_warmups_per_method': 1,
        'policy': 'resume missing planned calls only; preserve failed/time-stopped original segment; no result selection'}
    data['restart_warmups'] = []
    data['status'] = 'resuming'
    save(args.output / 'result.json', data)
    save(args.output / 'run_metadata.json', {'started_at': datetime.now().astimezone().isoformat(),
        'controller_pid': os.getpid(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'original_segment_preserved': True})
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + str(ROOT)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'QUIMB_NUM_PROCS'):
        env[key] = '1'
    command = [sys.executable, '-X', 'utf8', str(Path(__file__)), '--worker', '--source', str(args.source),
        '--output', str(args.output), '--seconds', str(args.seconds), '--memory-mib', str(args.memory_mib)]
    begin, peak, reason = time.monotonic(), 0, None
    with (args.output/'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                tree = psutil.Process(process.pid)
                children = tree.children(recursive=True)
                rss = tree.memory_info().rss + sum(child.memory_info().rss for child in children)
                peak = max(peak, rss)
                if rss > args.memory_mib * 2**20:
                    reason = 'process_tree_memory_budget_exceeded'
                elif psutil.virtual_memory().available < 1000 * 2**20:
                    reason = 'system_available_memory_below_1000_MiB'
                elif time.monotonic()-begin > args.seconds:
                    reason = 'wall_clock_budget_exceeded'
                if reason:
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.2)
        code = process.wait()
    final = json.loads((args.output/'result.json').read_text(encoding='utf-8'))
    passed = code == 0 and reason is None and final['status'] == 'pass'
    report = {'status': 'pass' if passed else 'incomplete_or_failed', 'exit_code': code, 'stop_reason': reason,
        'elapsed_s': time.monotonic()-begin, 'peak_process_tree_rss_mib': peak/2**20,
        'calibration_only': False, 'original_segment_exit_code': summary['exit_code'],
        'original_segment_stop_reason': summary['stop_reason'], 'segmented_run_disclosed': True,
        'formal_30q_acceptance_complete': False,
        'budgets': {'seconds': args.seconds, 'memory_mib': args.memory_mib}}
    save(args.output/'summary.json', report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seconds', type=float, default=5400)
    parser.add_argument('--memory-mib', type=float, default=1800)
    args = parser.parse_args()
    worker(args) if args.worker else suite(args)
