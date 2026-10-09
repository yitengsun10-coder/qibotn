"""Full exact30 supplementary evaluation; never replaces original hard30.

Each predeclared input gets an independent full-state oracle, official baseline
thread calibration, and three fresh measured repetitions of three methods.
Planning is the default. All three input results, including failures, persist.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ('upstream_typo_only', 'integrated', 'guarded_scaled_candidate')
CORE = [ROOT/'src/qibotn/backends'/name for name in (
    'quimb.py', '_exact_numpy.py', '_structured_numpy.py', '_linear_cnot.py',
    '_cnot_schedule.py', '_mixed_schedule.py')]
SOURCES = [*CORE, Path(__file__), *[HERE/name for name in (
    'candidate_guarded_scaled.py', 'supplement30_workloads.py',
    'profile_fixed30_bounded.py', 'upstream_reference.py')],
    ROOT/'benchmarks/asc_large_cpu/benchmark.py']


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def check_state(norm_oracle, norm_actual, overlap):
    if not all(math.isfinite(v) for v in [norm_oracle, norm_actual, overlap.real, overlap.imag]):
        raise ValueError('Nonfinite full-state comparison')
    if min(norm_oracle, norm_actual) <= 0:
        raise ValueError('Zero norm')
    fidelity = abs(overlap)**2/(norm_oracle*norm_actual)
    distance_sq = max(0., 2-2*abs(overlap)/math.sqrt(norm_oracle*norm_actual))
    if max(abs(norm_oracle-1), abs(norm_actual-1), abs(fidelity-1), distance_sq) > 1e-9:
        raise ValueError('Independent complete-state validation failed')
    return {'oracle_norm_squared': norm_oracle, 'mps_norm_squared': norm_actual,
            'fidelity_vs_independent_block_sum': fidelity,
            'phase_aligned_normalized_l2_squared': distance_sq,
            'full_state_comparison_pass': True,
            'scope': 'all 30 physical sites and all gates, untruncated independent block-sum oracle; not a 2**30 dense vector'}


def summarize(rows):
    expected = [(method, repeat) for repeat in range(4)
                for method in (METHODS if repeat%2 == 0 else METHODS[::-1])]
    if [(r['method'], r['repeat']) for r in rows] != expected:
        raise ValueError('Incomplete or reordered planned timings')
    if any(not math.isfinite(r['runtime_s']+r['abs_error']) or r['runtime_s'] <= 0
           or r['abs_error'] > 1e-10 for r in rows):
        raise ValueError('Invalid timing/correctness')
    samples = {m: [r['runtime_s'] for r in rows if r['method'] == m and r['repeat'] > 0] for m in METHODS}
    means = {m: statistics.mean(v) for m, v in samples.items()}
    return {'means_s': means, 'stdev_s': {m: statistics.stdev(v) for m, v in samples.items()},
            'speedups_vs_tuned_official': {m: means[METHODS[0]]/means[m] for m in METHODS[1:]},
            'candidate_speedup_vs_integrated': means['integrated']/means['guarded_scaled_candidate']}


def worker(args):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_revision_cpu.candidate_guarded_scaled import make_backend, prepare
    from benchmarks.asc_revision_cpu.supplement30_workloads import CONFIGURATIONS, BlockSumOracle, build, input_bytes
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_large_cpu.benchmark import make_observable
    declaration = json.loads((args.output.parent/'summary.json').read_text(encoding='utf-8'))
    for name, digest in declaration['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed after predeclaration')
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:16])
    seed = dict(CONFIGURATIONS)[args.kind]
    circuit = build(args.kind, seed)
    encoded = input_bytes(circuit)
    specification = next(x for x in declaration['workloads'] if x['kind'] == args.kind)
    if hashlib.sha256(encoded).hexdigest() != specification['input_sha256']:
        raise ValueError('Predeclared input changed')
    (args.output/'input.json').write_bytes(encoded)
    reference_class, reference = load_reference(args.output)
    methods = {METHODS[0]: reference_class(), METHODS[1]: QuimbNumpyBackend(), METHODS[2]: make_backend()}
    for backend in methods.values():
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    result = {**specification, 'status': 'independent_oracle', 'reference': reference,
              'full_state': {}, 'baseline_thread_calibration': [], 'rows': [],
              'warmup_repeats': 1, 'measured_repeats': 3,
              'formal_supplementary_30q_complete': False, 'original30_complete': False,
              'all_contest_tasks_complete': False, 'affinity': psutil.Process().cpu_affinity(),
              'timing_scope': 'new conversion/compiler + every gate + original four ZZ; excludes imports, oracle, separate full-state checks'}
    path = args.output/'result.json'
    save(path, result)
    labels, sites, coefficients = make_observable(30)
    with threadpool_limits(limits=1):
        oracle = BlockSumOracle.from_circuit(circuit)
        norm_oracle = oracle.expectation().real
        exact_value = sum(coefficient*oracle.expectation(word, support).real
                          for word, support, coefficient in zip(labels, sites, coefficients))/norm_oracle
        result.update(independent_observable=exact_value, oracle_norm_squared=norm_oracle,
                      oracle_gates_applied=oracle.gates_applied, status='baseline_thread_calibration')
        save(path, result)
    # Baseline gets the best measured configuration, not an inflated denominator.
    # All three methods subsequently use this SAME thread count and affinity.
    for threads in [1, 4, 8]:
        with threadpool_limits(limits=threads):
            begin = time.perf_counter()
            value = float(methods[METHODS[0]].exp_value_observable_symbolic(circuit, labels, sites, coefficients, 30))
            elapsed = time.perf_counter()-begin
        if abs(value-exact_value) > 1e-10 or not math.isfinite(value):
            raise ValueError('Calibration observable failure')
        result['baseline_thread_calibration'].append({'threads': threads, 'runtime_s': elapsed,
            'value': value, 'abs_error': abs(value-exact_value), 'not_a_formal_repeat': True})
        save(path, result)
        print(json.dumps({'kind': args.kind, 'calibration_threads': threads, 'runtime_s': elapsed}), flush=True)
    threads = min(result['baseline_thread_calibration'], key=lambda r:r['runtime_s'])['threads']
    with threadpool_limits(limits=threads):
        result.update(threads=threads, threadpools=threadpool_info())
        if any(p['num_threads'] != threads for p in result['threadpools'] if p['user_api'] == 'blas'):
            raise ValueError('Thread configuration not effective')
        for method, backend in methods.items():
            result['status'] = 'full_state_'+method
            save(path, result)
            if method == METHODS[0]:
                state = backend._qibo_circuit_to_quimb(circuit, quimb_circuit_type=backend.circuit_ansatz,
                    gate_opts={'max_bond': None, 'cutoff': 0}).psi
                details = {'engine': 'pinned_official'}
            elif method == METHODS[1]:
                state, details = backend._prepare_optimized_mps(circuit)
            else:
                state, details = prepare(backend, circuit)
            overlap, evidence = oracle.overlap_mps(state)
            result['full_state'][method] = {**check_state(norm_oracle, evidence['mps_norm_squared'], overlap),
                **evidence, 'overlap_real': overlap.real, 'overlap_imag': overlap.imag,
                'max_bond_observed': state.max_bond(), 'preparation': details}
            save(path, result)
            del state
            gc.collect()
        del oracle
        gc.collect()
        for repeat in range(4):
            for method in (METHODS if repeat%2 == 0 else METHODS[::-1]):
                result['status'] = f'timing_{method}_{repeat}'
                save(path, result)
                begin = time.perf_counter()
                value = float(methods[method].exp_value_observable_symbolic(circuit, labels, sites, coefficients, 30))
                row = {'method': method, 'repeat': repeat, 'runtime_s': time.perf_counter()-begin,
                       'value': value, 'abs_error': abs(value-exact_value)}
                result['rows'].append(row)
                save(path, result)
                print(json.dumps({'kind': args.kind, **row}), flush=True)
                if not math.isfinite(value) or row['abs_error'] > 1e-10:
                    raise ValueError('Formal observable failure')
        if input_bytes(circuit) != encoded:
            raise ValueError('Input mutated')
        result.update(summarize(result['rows']))
        result.update(status='pass', formal_supplementary_30q_complete=True)
        save(path, result)


def suite(args):
    import psutil
    from benchmarks.asc_revision_cpu.profile_fixed30_bounded import effective_memory
    from benchmarks.asc_revision_cpu.supplement30_workloads import CONFIGURATIONS, build, metadata
    if args.output.exists():
        raise ValueError('Fresh destination required')
    args.output.mkdir(parents=True)
    report = {'recorded_at': datetime.now().astimezone().isoformat(), 'status': 'plan_only',
        'execute_requested': args.execute, 'original30_complete': False, 'all_contest_tasks_complete': False,
        'supplement30_suite_complete': False,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
        'workloads': [metadata(build(kind, seed), kind, seed) for kind, seed in CONFIGURATIONS],
        'scope': 'predeclared extra profiles, user-authorized; preserve original high-entanglement30 as incomplete',
        'budgets_per_case': {'seconds': args.seconds, 'rss_mib': args.memory_mib, 'reserve_mib': 2048},
        'environment': {'python': sys.version, 'platform': platform.platform(),
            'affinity': psutil.Process().cpu_affinity(), 'memory': effective_memory(psutil.virtual_memory().available),
            'versions': {n: importlib.metadata.version(n) for n in ['qibo','quimb','numpy','scipy','psutil','threadpoolctl']}},
        'cases': []}
    for source in SOURCES:
        destination = args.output/'source_snapshot'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    save(args.output/'summary.json', report)
    if not args.execute:
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+str(ROOT)
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','QUIMB_NUM_PROCS']:
        env[name] = '1'
    for kind, _ in CONFIGURATIONS:
        if effective_memory(psutil.virtual_memory().available)['available_bytes'] < (args.memory_mib+2048)*2**20:
            raise ValueError('Insufficient effective memory; no new worker started')
        destination = args.output/kind
        destination.mkdir()
        command = [sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',
                   '--output',str(destination),'--kind',kind]
        begin, peak, reason = time.monotonic(), 0, None
        with (destination/'run.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            tree = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    children = tree.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in [tree,*children])
                    peak = max(peak,rss)
                    if rss > args.memory_mib*2**20:
                        reason='process_tree_memory_budget'
                    elif effective_memory(psutil.virtual_memory().available)['available_bytes'] < 2048*2**20:
                        reason='effective_memory_reserve'
                    elif time.monotonic()-begin > args.seconds:
                        reason='wall_time_budget'
                    if reason:
                        for child in children:
                            try: child.kill()
                            except psutil.NoSuchProcess: pass
                        process.kill()
                        break
                except psutil.NoSuchProcess: pass
                time.sleep(.1)
            code=process.wait()
        path=destination/'result.json'
        data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        report['cases'].append({'kind':kind,'exit_code':code,'stop_reason':reason,
            'wall_s':time.monotonic()-begin,'peak_process_tree_rss_bytes':peak,
            'pass':code==0 and reason is None and data.get('status')=='pass'})
        save(args.output/'summary.json',report)
        print(json.dumps(report['cases'][-1]),flush=True)
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                  for name,digest in report['source_hashes'].items())
    passed=unchanged and len(report['cases'])==3 and all(x['pass'] for x in report['cases'])
    report.update(status='pass_supplementary_only' if passed else 'incomplete_or_failed',
                  source_unchanged=unchanged,supplement30_suite_complete=passed)
    save(args.output/'summary.json',report)
    if not passed: raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--kind',choices=['chain','ring','nonlocal_mixed'])
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--memory-mib',type=float,default=4096)
    args=parser.parse_args()
    if not all(math.isfinite(v) and v>0 for v in [args.seconds,args.memory_mib]):
        parser.error('Finite positive budgets required')
    if args.worker and args.kind is None: parser.error('Worker requires a kind')
    worker(args) if args.worker else suite(args)
