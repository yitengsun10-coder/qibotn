"""Bounded kernel calibration, NOT a circuit result or contest speedup.

Use the observed exact QR shapes, identical seeded complex128 matrices, fresh
sequential workers and actual BLAS pool checks. One sample per configuration
is exploratory only. Never replaces the official end-to-end baseline.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def worker(args):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits, threadpool_info
    from qibotn.backends._exact_numpy import _reduced_qr
    rng = np.random.default_rng(20261093)
    matrix = rng.normal(size=(args.rows, args.columns)) + 1j*rng.normal(size=(args.rows, args.columns))
    digest = hashlib.sha256(memoryview(matrix).cast('B')).hexdigest()
    probes = rng.normal(size=(args.columns, 4)) + 1j*rng.normal(size=(args.columns, 4))
    data = {'status': 'running_kernel_calibration', 'shape': list(matrix.shape),
            'dtype': str(matrix.dtype), 'threads_requested': args.threads,
            'input_sha256': digest, 'affinity': psutil.Process().cpu_affinity(),
            'formal_30q_acceptance_complete': False, 'publishable_speedup': None}
    with threadpool_limits(limits=args.threads):
        data['threadpools'] = threadpool_info()
        if any(p['num_threads'] != args.threads for p in data['threadpools'] if p['user_api'] == 'blas'):
            raise ValueError('Requested BLAS limit not effective')
        save(args.output / 'result.json', data)
        begin = time.perf_counter()
        q, r = _reduced_qr(matrix)
        elapsed = time.perf_counter()-begin
        expected = matrix @ probes
        residual = float(np.linalg.norm(q @ (r @ probes)-expected)/np.linalg.norm(expected))
        orthogonality = float(np.linalg.norm(q.conj().T @ (q @ probes)-probes)/np.linalg.norm(probes))
        if hashlib.sha256(memoryview(matrix).cast('B')).hexdigest() != digest:
            raise ValueError('QR modified caller input')
        if not np.isfinite(residual+orthogonality) or max(residual, orthogonality) > 1e-10:
            raise ValueError('Random-projection QR check failed')
        data.update(status='pass_kernel_calibration_not_acceptance', runtime_s=elapsed,
                    projection_relative_residual=residual,
                    projection_orthogonality_residual=orthogonality,
                    numerical_check_scope='four deterministic random projections, not full matrix or circuit validation')
        save(args.output / 'result.json', data)


def suite(args):
    import psutil
    from benchmarks.asc_revision_cpu.profile_fixed30_bounded import effective_memory
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    sources = [Path(__file__), ROOT/'src/qibotn/backends/_exact_numpy.py',
               ROOT/'benchmarks/asc_revision_cpu/profile_fixed30_bounded.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for source in sources:
        destination = args.output/'source_snapshot'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    report = {'status': 'prepared_kernel_calibration', 'recorded_at': datetime.now().astimezone().isoformat(),
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None,
        'scope': 'one exploratory exact QR sample per shape/thread setting; never an end-to-end speedup',
        'budgets_per_job': {'seconds': args.seconds, 'rss_mib': args.memory_mib, 'reserve_mib': 2048},
        'source_hashes': hashes, 'python': sys.version,
        'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'threadpoolctl', 'psutil']},
        'memory': effective_memory(psutil.virtual_memory().available), 'rows': []}
    save(args.output/'summary.json', report)
    if not args.execute:
        return
    allowed = psutil.Process().cpu_affinity()
    if max(args.thread_options) > len(allowed):
        raise ValueError('More requested threads than allowed CPUs')
    shapes = [(64, 32)] if args.smoke_test else [(4096, 1024), (8192, 2048)]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+str(ROOT)
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        environment[name] = '1'
    for rows, columns in shapes:
        for threads in args.thread_options:
            if effective_memory(psutil.virtual_memory().available)['available_bytes'] < (args.memory_mib+2048)*2**20:
                report['status'] = 'not_started_memory_reserve'
                save(args.output/'summary.json', report)
                return
            destination = args.output/f'qr_{rows}_{columns}_t{threads}'
            destination.mkdir()
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker',
                       '--output', str(destination), '--rows', str(rows), '--columns', str(columns),
                       '--threads', str(threads)]
            begin, peak, reason = time.monotonic(), 0, None
            with (destination/'run.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                tree = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        children = tree.children(recursive=True)
                        rss = sum(p.memory_info().rss for p in [tree, *children])
                        peak = max(peak, rss)
                        if rss > args.memory_mib*2**20:
                            reason = 'process_tree_memory_budget'
                        elif effective_memory(psutil.virtual_memory().available)['available_bytes'] < 2048*2**20:
                            reason = 'effective_memory_reserve'
                        elif time.monotonic()-begin > args.seconds:
                            reason = 'wall_time_budget'
                        if reason:
                            for child in children:
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            process.kill()
                            break
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.1)
                code = process.wait()
            result = destination/'result.json'
            entry = {'shape': [rows, columns], 'threads': threads, 'exit_code': code,
                     'stop_reason': reason, 'wall_s': time.monotonic()-begin,
                     'peak_process_tree_rss_bytes': peak,
                     'result': json.loads(result.read_text()) if result.exists() else None}
            report['rows'].append(entry)
            save(args.output/'summary.json', report)
            print(json.dumps({k: entry[k] for k in ['shape', 'threads', 'exit_code', 'stop_reason']}), flush=True)
    report.update(status='finished_kernel_calibration_not_acceptance', source_unchanged=all(
        hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items()))
    save(args.output/'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--thread-options', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--rows', type=int)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--seconds', type=float, default=45)
    parser.add_argument('--memory-mib', type=float, default=4096)
    args = parser.parse_args()
    if (not all(math.isfinite(v) and v > 0 for v in [args.seconds, args.memory_mib])
            or min(args.thread_options) < 1 or len(set(args.thread_options)) != len(args.thread_options)):
        parser.error('Positive budgets and thread counts required')
    if args.worker and any(v is None or v < 1 for v in [args.rows, args.columns, args.threads]):
        parser.error('Worker requires positive dimensions and threads')
    worker(args) if args.worker else suite(args)
