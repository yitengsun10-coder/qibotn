"""Predeclared exact 30q acceptance entry; planning never simulates a circuit.

The existing 30q/9998-gate/depth449 input is unchanged. A CPU machine with
sufficient RAM is required for execution. No truncation, dense 30q oracle,
candidate retry, result selection, or automatic cloud resource creation.
Full-state checking streams a pinned official MPS from local NPY files. This
is a complete state comparison against that implementation, NOT an independent
30q dense oracle. Independent dense checks remain separate small-scale evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
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
CORE = tuple(ROOT / 'src/qibotn/backends' / name for name in (
    'quimb.py', '_exact_numpy.py', '_linear_cnot.py', '_structured_numpy.py',
    '_cnot_schedule.py', '_mixed_schedule.py'))
SOURCES = (*CORE, Path(__file__), HERE / 'capacity_plan.py',
           HERE / 'upstream_reference.py', ROOT / 'benchmarks/asc_large_cpu/benchmark.py')
METHODS = ('upstream_typo_only', 'integrated')
QASM_SHA256 = '1ea5c95ebdcc651a1291dadfb7a193009377732bba8d03a01ac37ef7b82b37d4'


def save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def make_plan(total_bytes, available_bytes, disk_free_bytes, process_budget_bytes):
    from benchmarks.asc_revision_cpu.capacity_plan import exact_mps_bounds
    bound = exact_mps_bounds(30)
    # Conservative admission policy, not a measured peak or universal 30q need.
    # Streaming the reference avoids retaining two complete states in RAM.
    reserve = 2 * bound['largest_merged_tensor_bytes'] + 2 * 2**30
    allowance = bound['full_rank_mps_arrays_bytes'] + reserve
    disk_needed = bound['full_rank_mps_arrays_bytes'] + 2 * 2**30
    ready = (process_budget_bytes >= allowance and
             available_bytes >= process_budget_bytes + 2 * 2**30 and
             disk_free_bytes >= disk_needed)
    return {'recorded_at': datetime.now().astimezone().isoformat(),
        'status': 'plan_only_not_acceptance', 'formal_30q_acceptance_complete': False,
        'publishable_speedup': None, 'qubits': 30, 'layers': 112, 'gates': 9998,
        'depth': 449, 'qasm_sha256': QASM_SHA256, 'precision': 'complex128',
        'cutoff': 0, 'max_bond': None, 'warmup_repeats': 1, 'measured_repeats': 3,
        'observable': 'original four ZZ terms, equal coefficients',
        'host_total_memory_bytes': total_bytes, 'host_available_memory_bytes': available_bytes,
        'disk_free_bytes': disk_free_bytes, 'process_budget_bytes': process_budget_bytes,
        'dimension_upper_bound': bound, 'conservative_process_admission_bytes': allowance,
        'conservative_disk_admission_bytes': disk_needed, 'resources_admitted': bool(ready),
        'note': 'Full-rank planning policy is conservative, not observed memory and not proof that every 30q circuit needs this RAM.',
        'correctness_scope': 'complete MPS overlap against pinned official zero-truncation CPU implementation; no independent dense 30q oracle',
        'policy': 'fixed input and all gates; same threads/affinity; original four ZZ; no answer-based selection; no precision reduction'}


def site_array(state, site, np):
    tensor = state[site]
    left = [state.bond(site - 1, site)] if site else []
    right = [state.bond(site, site + 1)] if site + 1 < state.L else []
    data = np.asarray(tensor.transpose(*left, *right, state.site_ind(site)).data)
    if data.dtype != np.dtype('complex128'):
        raise ValueError('Only complex128 states qualify')
    return data.reshape((data.shape[0] if left else 1,
                         data.shape[len(left)] if right else 1, 2))


def export_state(state, destination, np):
    destination.mkdir()
    records = []
    for site in range(state.L):
        array = site_array(state, site, np)
        path = destination / f'site_{site:02d}.npy'
        np.save(path, array, allow_pickle=False)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records.append({'file': path.name, 'shape': list(array.shape), 'sha256': digest})
    save(destination / 'manifest.json', {'sites': records, 'dtype': 'complex128', 'logical_site_order': list(range(state.L))})


def streamed_overlap(reference_dir, state, np):
    """Contract <reference|state> without a dense vector or a second full MPS.

    Explicit physical-channel transfer is separate from the optimized
    observable evaluator. State physical sites must be restored logical order.
    """
    manifest = json.loads((reference_dir / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['sites']) != state.L:
        raise ValueError('Reference site count changed')
    environment = np.ones((1, 1), dtype=np.complex128)
    for site, record in enumerate(manifest['sites']):
        path = reference_dir / record['file']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != record['sha256']:
            raise ValueError('Reference tensor bytes changed')
        reference = np.load(path, mmap_mode='r', allow_pickle=False)
        actual = site_array(state, site, np)
        if list(reference.shape) != record['shape'] or reference.dtype != np.dtype('complex128'):
            raise ValueError('Reference shape/precision changed')
        if reference.shape[0] != environment.shape[0] or actual.shape[0] != environment.shape[1]:
            raise ValueError('MPS bond mismatch')
        result = None
        for physical in range(2):
            term = reference[:, :, physical].conj().T @ environment @ actual[:, :, physical]
            if result is None:
                result = term
            else:
                result += term
        environment = result
        del reference, actual, term
    if environment.shape != (1, 1):
        raise ValueError('Open MPS boundary mismatch')
    return complex(environment[0, 0])


def validate_state(norm_reference, norm_actual, overlap):
    values = (norm_reference, norm_actual, overlap.real, overlap.imag)
    if not all(map(math.isfinite, values)) or min(norm_reference, norm_actual) <= 0:
        raise ValueError('Non-finite or zero-norm state')
    fidelity = abs(overlap)**2 / (norm_reference * norm_actual)
    distance_sq = max(0., 2 - 2 * abs(overlap) / math.sqrt(norm_reference * norm_actual))
    if (abs(norm_reference - 1) > 1e-9 or abs(norm_actual - 1) > 1e-9 or
            abs(fidelity - 1) > 1e-9 or distance_sq > 1e-9):
        raise ValueError('Complete exact state comparison failed')
    return {'reference_norm_squared': norm_reference, 'integrated_norm_squared': norm_actual,
            'fidelity_vs_pinned_official': fidelity, 'normalized_phase_aligned_l2_squared': distance_sq,
            'full_state_comparison_pass': True, 'independent_dense_30q_oracle': False}


def summarize_rows(rows):
    expected = [(method, repeat) for repeat in range(4)
                for method in (METHODS if repeat % 2 == 0 else METHODS[::-1])]
    if [(row['method'], row['repeat']) for row in rows] != expected:
        raise ValueError('Missing, extra or out-of-order planned rows')
    reference_value = rows[0]['value']
    for row in rows:
        if (not all(math.isfinite(row[k]) for k in ('runtime_s', 'value')) or
                row['runtime_s'] <= 0 or abs(row['value'] - reference_value) > 1e-10):
            raise ValueError('Observable correctness or runtime invalid')
    samples = {method: [r['runtime_s'] for r in rows if r['method'] == method and r['repeat'] > 0]
               for method in METHODS}
    means = {method: statistics.mean(values) for method, values in samples.items()}
    return {'means_s': means, 'stdev_s': {m: statistics.stdev(v) for m, v in samples.items()},
            'speedup': means[METHODS[0]] / means[METHODS[1]],
            'max_observable_difference_vs_official_warmup': max(abs(r['value']-reference_value) for r in rows)}


def worker(args):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit, make_observable
    if args.affinity:
        psutil.Process().cpu_affinity(args.affinity)
    metadata = json.loads((args.output / 'run_metadata.json').read_text(encoding='utf-8'))
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in metadata['source_hashes'].items()):
        raise ValueError('Source changed after snapshot')
    circuit = build_circuit(Workload(30, 112))
    qasm = circuit.to_qasm()
    if (len(circuit.queue), circuit.depth, hashlib.sha256(qasm.encode()).hexdigest()) != (9998, 449, QASM_SHA256):
        raise ValueError('Predeclared 30q input changed')
    (args.output / 'input.qasm').write_text(qasm, encoding='utf-8')
    reference_class, reference = load_reference(args.output)
    methods = {METHODS[0]: reference_class(), METHODS[1]: QuimbNumpyBackend()}
    for backend in methods.values():
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    labels, sites, coefficients = make_observable(30)
    data = {**metadata, 'status': 'full_state_reference', 'reference': reference,
            'rows': [], 'formal_30q_acceptance_complete': False,
            'timing_scope': 'full conversion/compiler + all gates + original four ZZ, excludes imports and separate full-state checks'}
    path = args.output / 'result.json'
    with threadpool_limits(limits=1):
        data['threadpools'] = threadpool_info()
        if any(pool['num_threads'] != 1 for pool in data['threadpools'] if pool['user_api'] == 'blas'):
            raise ValueError('Both paths must actually use one BLAS thread')
        save(path, data)
        official = methods[METHODS[0]]
        state = official._qibo_circuit_to_quimb(circuit, quimb_circuit_type=official.circuit_ansatz,
                    gate_opts={'max_bond': None, 'cutoff': 0}).psi
        norm_reference = float(np.real(state.H @ state))
        data['reference_max_bond'] = state.max_bond()
        export_state(state, args.output / 'reference_mps', np)
        del state
        gc.collect()
        data['status'] = 'full_state_integrated'
        save(path, data)
        state, preparation = methods[METHODS[1]]._prepare_optimized_mps(circuit)
        norm_actual = float(np.real(state.H @ state))
        overlap = streamed_overlap(args.output / 'reference_mps', state, np)
        data['full_state'] = {**validate_state(norm_reference, norm_actual, overlap),
                              'integrated_max_bond': state.max_bond(), 'preparation': preparation}
        del state
        gc.collect()
        save(path, data)
        for repeat in range(4):
            for method in (METHODS if repeat % 2 == 0 else METHODS[::-1]):
                data['status'] = f'timing_{method}_repeat_{repeat}'
                save(path, data)
                begin = time.perf_counter()
                value = float(methods[method].exp_value_observable_symbolic(circuit, labels, sites, coefficients, 30))
                elapsed = time.perf_counter() - begin
                row = {'method': method, 'repeat': repeat, 'runtime_s': elapsed, 'value': value}
                data['rows'].append(row)
                save(path, data)
                print(json.dumps(row), flush=True)
                if not math.isfinite(value) or abs(value-data['rows'][0]['value']) > 1e-10:
                    raise ValueError('Timed observable disagrees with pinned official')
        if circuit.to_qasm() != qasm:
            raise ValueError('Original input changed')
        data.update(summarize_rows(data['rows']))
        data.update(status='pass', formal_30q_acceptance_complete=True,
                    seven_x_achieved=data['speedup'] >= 7, all_contest_tasks_complete=False)
        save(path, data)


def suite(args):
    import psutil
    if args.output.exists():
        raise ValueError('Fresh directory required; old evidence is preserved')
    args.output.mkdir(parents=True)
    memory = psutil.virtual_memory()
    plan = make_plan(memory.total, memory.available, shutil.disk_usage(args.output).free, args.memory_gib * 2**30)
    save(args.output / 'resource_plan.json', plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not plan['resources_admitted']:
        save(args.output / 'summary.json', {**plan, 'status': 'not_started_resource_preflight', 'exit_code': None})
        raise SystemExit(2)
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        command = process.info['cmdline'] or []
        if '--worker' in command and any('pressure_chain.py' in item for item in command):
            raise ValueError('A formal pressure timing worker is active; do not interfere')
    metadata = {**plan, 'status': 'started', 'affinity': args.affinity or psutil.Process().cpu_affinity(),
        'threads': 1, 'controller_pid': os.getpid(), 'source_hashes': {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}}
    save(args.output / 'run_metadata.json', metadata)
    for source in SOURCES:
        shutil.copyfile(source, args.output / (source.stem + '_snapshot.py'))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'QUIMB_NUM_PROCS'):
        env[key] = '1'
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', '--output', str(args.output)]
    if args.affinity:
        command += ['--affinity', *map(str, args.affinity)]
    begin, peak, reason = time.monotonic(), 0, None
    with (args.output / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                tree = psutil.Process(process.pid)
                children = tree.children(recursive=True)
                peak = max(peak, sum(p.memory_info().rss for p in [tree, *children]))
                if peak > args.memory_gib * 2**30:
                    reason = 'process_tree_memory_budget_exceeded'
                elif psutil.virtual_memory().available < 2 * 2**30:
                    reason = 'system_available_memory_below_2_GiB'
                elif time.monotonic() - begin > args.seconds:
                    reason = 'wall_clock_budget_exceeded'
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
    path = args.output / 'result.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    passed = code == 0 and reason is None and data.get('status') == 'pass'
    save(args.output / 'summary.json', {'status': 'pass' if passed else 'incomplete_or_failed',
        'exit_code': code, 'stop_reason': reason, 'elapsed_s': time.monotonic() - begin,
        'peak_process_tree_rss_bytes': peak, 'formal_30q_acceptance_complete': passed,
        'speedup': data.get('speedup') if passed else None, 'all_contest_tasks_complete': False})
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Never implied by planning')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--affinity', nargs='+', type=int)
    parser.add_argument('--memory-gib', type=float, default=96)
    parser.add_argument('--seconds', type=float, default=7200)
    args = parser.parse_args()
    if args.memory_gib <= 0 or args.seconds <= 0:
        parser.error('Positive resource/time limits required')
    worker(args) if args.worker else suite(args)
