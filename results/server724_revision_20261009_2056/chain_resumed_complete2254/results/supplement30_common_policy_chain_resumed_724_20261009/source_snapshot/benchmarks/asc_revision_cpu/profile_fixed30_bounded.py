"""Bounded, instrumented diagnosis of the unchanged exact 30q input.

Never a formal speed measurement or an acceptance result. Profiling adds
overhead; partial paths and unlike progress counters must not form a ratio.
Production files stay unchanged. No cloud resource is created by this tool.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata as package_metadata
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QASM_SHA256 = '1ea5c95ebdcc651a1291dadfb7a193009377732bba8d03a01ac37ef7b82b37d4'
METHODS = ('upstream_typo_only', 'integrated')
RESEARCH_METHOD = 'guarded_scaled_candidate'
CORE = tuple(ROOT / 'src/qibotn/backends' / name for name in (
    'quimb.py', '_exact_numpy.py', '_linear_cnot.py', '_structured_numpy.py',
    '_cnot_schedule.py', '_mixed_schedule.py'))


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def effective_memory(host_available, cgroup_root=Path('/sys/fs/cgroup')):
    """Respect container RAM, not the often much larger host MemAvailable."""
    for limit_name, used_name in (
        ('memory.max', 'memory.current'),
        ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')):
        try:
            limit_text = (cgroup_root / limit_name).read_text().strip()
            if limit_text == 'max':
                continue
            limit = int(limit_text)
            used = int((cgroup_root / used_name).read_text().strip())
            if not 0 < limit < 2**60 or used < 0:
                continue
            return {'available_bytes': min(host_available, max(0, limit - used)),
                    'cgroup_limit_bytes': limit, 'cgroup_used_bytes': used,
                    'host_available_bytes': host_available}
        except (OSError, ValueError):
            continue
    return {'available_bytes': host_available, 'cgroup_limit_bytes': None,
            'cgroup_used_bytes': None, 'host_available_bytes': host_available}


class ProbeStop(RuntimeError):
    pass


class Recorder:
    """Nested stage accounting; keeps scalar metadata, never tensor copies."""
    def __init__(self, guard=lambda extra: None, observe=lambda: None):
        self.guard, self.observe = guard, observe
        self.stages, self.stack, self.patches = {}, [], []

    def patch(self, owner, attribute, label, before=None, after=None):
        original = getattr(owner, attribute)
        owned = attribute in vars(owner)
        raw_original = vars(owner).get(attribute)
        self.patches.append((owner, attribute, raw_original, owned))

        def wrapped(*args, **kwargs):
            metadata = before(args, kwargs) if before else {}
            self.guard(metadata.get('estimated_extra_bytes', 0))
            row = self.stages.setdefault(label, {'calls': 0, 'completed_calls': 0,
                'inclusive_s': 0., 'exclusive_s': 0., 'max_completed_s': 0.,
                'max_input_bytes': 0, 'largest_input_shape': None})
            row['calls'] += 1
            if metadata.get('input_bytes', 0) > row['max_input_bytes']:
                row['max_input_bytes'] = metadata['input_bytes']
                row['largest_input_shape'] = metadata.get('shape')
            frame = [label, time.perf_counter(), 0.]
            self.stack.append(frame)
            self.observe()
            completed = False
            try:
                result = original(*args, **kwargs)
                completed = True
                if after:
                    after(args, kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - frame[1]
                self.stack.pop()
                row['inclusive_s'] += elapsed
                row['exclusive_s'] += max(0., elapsed - frame[2])
                if completed:
                    row['completed_calls'] += 1
                    row['max_completed_s'] = max(row['max_completed_s'], elapsed)
                if self.stack:
                    self.stack[-1][2] += elapsed
                self.observe()
        setattr(owner, attribute, wrapped)

    def restore(self):
        for owner, attribute, original, owned in reversed(self.patches):
            if owned:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.patches.clear()


def worker(args):
    import numpy as np
    import psutil
    import quimb.tensor as qtn
    from qibo.backends import NumpyBackend
    from threadpoolctl import threadpool_info, threadpool_limits
    from qibotn.backends import _exact_numpy as exact, _structured_numpy as structured
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit, make_observable
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference

    process = psutil.Process()
    allowed = process.cpu_affinity()
    if args.threads > len(allowed):
        raise ValueError('Requested threads exceed allowed CPUs')
    affinity = allowed[:min(max(4, args.threads), len(allowed))]
    process.cpu_affinity(affinity)
    specification = Workload(6, 3) if args.smoke_test else Workload(30, 112)
    circuit = build_circuit(specification)
    qasm = circuit.to_qasm()
    digest = hashlib.sha256(qasm.encode()).hexdigest()
    if not args.smoke_test and (len(circuit.queue), circuit.depth, digest) != (9998, 449, QASM_SHA256):
        raise ValueError('Fixed 30q input changed')
    (args.output / 'input.qasm').write_text(qasm, encoding='utf-8')
    metadata = json.loads((args.output.parent / 'summary.json').read_text(encoding='utf-8'))
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
           for name, value in metadata['source_hashes'].items()):
        raise ValueError('Source changed after snapshot')
    if args.method == METHODS[0]:
        backend_class, reference = load_reference(args.output)
        backend = backend_class()
    elif args.method == RESEARCH_METHOD:
        from benchmarks.asc_revision_cpu.candidate_guarded_scaled import make_backend
        backend, reference = make_backend(), None
    else:
        backend, reference = QuimbNumpyBackend(), None
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    labels, sites, coefficients = make_observable(specification.qubits)
    begin = time.perf_counter()
    progress = {'completed_apply_gate_calls': 0, 'completed_apply_gate_raw_calls': 0,
                'max_bond_observed': 1, 'mps_array_bytes': 0, 'identity_snapshot_bytes': 0,
                'counter_note': 'API calls, not original input gates; fusion/nesting differ by path'}
    result = {'status': 'running', 'method': args.method, 'qubits': specification.qubits,
        'gates': len(circuit.queue), 'depth': circuit.depth, 'qasm_sha256': digest,
        'precision': 'complex128', 'cutoff': 0, 'max_bond': None, 'affinity': affinity,
        'threads_requested': args.threads,
        'reference': reference, 'instrumented_not_formal_timing': True,
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None,
        'smoke_test_only': args.smoke_test, 'progress': progress}
    last_write, last_state = 0., None
    milestones = []

    def guard(extra):
        memory = effective_memory(psutil.virtual_memory().available)
        rss = process.memory_info().rss
        if memory['available_bytes'] - extra < args.reserve_mib * 2**20:
            raise ProbeStop('effective_available_memory_or_projected_workspace_reserve')
        if rss + extra > args.memory_mib * 2**20:
            raise ProbeStop('process_rss_or_projected_workspace_budget')
        if time.perf_counter() - begin > args.seconds:
            raise ProbeStop('instrumented_worker_time_budget')

    def observe(force=False):
        nonlocal last_write
        now = time.perf_counter()
        if not force and now - last_write < .5:
            return
        result.update(instrumented_elapsed_s=now-begin, stages=recorder.stages,
                      in_flight_stages=[frame[0] for frame in recorder.stack],
                      memory=effective_memory(psutil.virtual_memory().available),
                      current_process_rss_bytes=process.memory_info().rss,
                      bond_growth_milestones=milestones)
        save(args.output / 'profile.json', result)
        last_write = now

    def after_gate(counter):
        def after(arguments, keywords):
            nonlocal last_state
            state = arguments[0]._psi
            last_state = state
            progress[counter] += 1
            bond = state.max_bond()
            progress['mps_array_bytes'] = sum(t.data.nbytes for t in state.tensor_map.values())
            progress['identity_snapshot_bytes'] = sum(
                item[1].nbytes for item in getattr(state, '_identity_gauges', {}).values())
            progress['analytic_gauge_shifts'] = getattr(state, 'analytic_gauge_shifts', 0)
            if bond > progress['max_bond_observed']:
                progress['max_bond_observed'] = bond
                milestones.append({**progress, 'instrumented_elapsed_s': time.perf_counter()-begin,
                                   'rss_bytes': process.memory_info().rss})
        return after

    def matrix_metadata(arguments, keywords):
        matrix = arguments[0]
        return {'input_bytes': matrix.nbytes, 'shape': list(matrix.shape),
                'estimated_extra_bytes': 4 * matrix.nbytes}

    def two_site_metadata(arguments, keywords):
        state, _, i, j = arguments[:4]
        left = state._layout(i)[1].shape[0]
        right = state._layout(j)[1].shape[1]
        size = 4 * left * right * 16
        return {'input_bytes': size, 'shape': [2*left, 2*right],
                'estimated_extra_bytes': 5 * size}

    recorder = Recorder(guard, observe)
    with threadpool_limits(limits=args.threads):
        result['threadpools'] = threadpool_info()
        if any(p['num_threads'] != args.threads for p in result['threadpools'] if p['user_api'] == 'blas'):
            raise ValueError('Requested BLAS limit not effective')
        try:
            recorder.patch(qtn.CircuitMPS, 'apply_gate', 'apply_gate_inclusive',
                           after=after_gate('completed_apply_gate_calls'))
            recorder.patch(qtn.CircuitMPS, 'apply_gate_raw', 'apply_gate_raw_inclusive',
                           after=after_gate('completed_apply_gate_raw_calls'))
            for cls, label in ((qtn.MatrixProductState, 'native'),
                               (exact.ExactKernelMPS, 'exact'),
                               (structured.StructuredExactKernelMPS, 'structured')):
                for attribute in ('left_canonize_site', 'right_canonize_site'):
                    recorder.patch(cls, attribute, label+'.'+attribute)
            if args.method == RESEARCH_METHOD:
                from benchmarks.asc_revision_cpu.candidate_guarded_scaled import GuardedScaledMPS
                for attribute in ('left_canonize_site', 'right_canonize_site'):
                    recorder.patch(GuardedScaledMPS, attribute, 'guarded_scaled.'+attribute)
            recorder.patch(qtn.MatrixProductState, 'gate_split', 'native.gate_split')
            recorder.patch(qtn.MatrixProductState, 'gate_split_', 'native.gate_split_inplace')
            recorder.patch(exact.ExactKernelMPS, '_two_site', 'exact.two_site', before=two_site_metadata)
            recorder.patch(structured.StructuredExactKernelMPS, '_two_site', 'structured.two_site', before=two_site_metadata)
            recorder.patch(exact, '_reduced_qr', 'exact.reduced_qr', before=matrix_metadata)
            recorder.patch(exact, '_left_isometric_factor', 'exact.left_isometric_factor', before=matrix_metadata)
            observe(force=True)
            value = float(backend.exp_value_observable_symbolic(
                circuit, labels, sites, coefficients, specification.qubits))
            if not math.isfinite(value):
                raise ValueError('Nonfinite observable')
            result.update(status='completed_instrumented_call_not_acceptance', observable=value)
        except ProbeStop as error:
            result.update(status='bounded_incomplete', stop_reason=str(error))
        finally:
            recorder.restore()
            observe(force=True)
        if args.smoke_test and result['status'] == 'completed_instrumented_call_not_acceptance':
            oracle = np.asarray(NumpyBackend().execute_circuit(circuit).state()).reshape(-1)
            actual = np.asarray(last_state.to_dense()).reshape(-1)
            result['smoke_full_state_max_abs_error'] = float(np.max(np.abs(actual-oracle)))
            if result['smoke_full_state_max_abs_error'] > 1e-10:
                raise ValueError('Instrumented smoke state changed')
            observe(force=True)
    return 0 if result['status'] == 'completed_instrumented_call_not_acceptance' else 2


def suite(args):
    import psutil
    if args.output.exists():
        raise ValueError('Use a new output directory; never overwrite prior evidence')
    args.output.mkdir(parents=True)
    sources = (*CORE, Path(__file__), HERE / 'upstream_reference.py',
               HERE / 'candidate_guarded_scaled.py',
               ROOT / 'benchmarks/asc_large_cpu/benchmark.py')
    report = {'recorded_at': datetime.now().astimezone().isoformat(),
        'status': 'plan_only', 'execute_requested': args.execute,
        'instrumented_not_formal_timing': True, 'formal_30q_acceptance_complete': False,
        'publishable_speedup': None, 'smoke_test_only': args.smoke_test,
        'methods': args.methods, 'threads_requested': args.threads,
        'budgets_per_path': {'seconds': args.seconds, 'process_tree_memory_mib': args.memory_mib,
                             'effective_memory_reserve_mib': args.reserve_mib},
        'workspace_guard_note': '4/5-array extra-space heuristic; conservative guard, not a measured allocation or proof of required RAM',
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'memory_at_start': effective_memory(psutil.virtual_memory().available),
        'environment': {'python': sys.version, 'python_executable': sys.executable,
            'platform': platform.platform(), 'host_logical_cpus': psutil.cpu_count(),
            'allowed_affinity': psutil.Process().cpu_affinity(),
            'packages': {name: package_metadata.version(name) for name in
                ('qibo', 'quimb', 'numpy', 'scipy', 'cotengra', 'autoray', 'psutil', 'threadpoolctl')}},
        'paths': []}
    for source in sources:
        destination = args.output / 'source_snapshot' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    save(args.output / 'summary.json', report)
    if not args.execute:
        return 0
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'QUIMB_NUM_PROCS'):
        environment[key] = '1'
    for method in args.methods:
        available = effective_memory(psutil.virtual_memory().available)
        if available['available_bytes'] < (args.memory_mib+args.reserve_mib)*2**20:
            report['paths'].append({'method': method, 'status': 'not_started_insufficient_memory',
                                    'memory': available})
            save(args.output / 'summary.json', report)
            continue
        destination = args.output / method
        destination.mkdir()
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker',
            '--output', str(destination), '--method', method, '--seconds', str(args.seconds),
            '--memory-mib', str(args.memory_mib), '--reserve-mib', str(args.reserve_mib),
            '--threads', str(args.threads)]
        if args.smoke_test:
            command.append('--smoke-test')
        begin, peak, reason = time.monotonic(), 0, None
        with (destination / 'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            tree = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    children = tree.children(recursive=True)
                    rss = tree.memory_info().rss + sum(c.memory_info().rss for c in children)
                    peak = max(peak, rss)
                    memory = effective_memory(psutil.virtual_memory().available)
                    if rss > args.memory_mib*2**20:
                        reason = 'supervisor_process_tree_rss_budget'
                    elif memory['available_bytes'] < args.reserve_mib*2**20:
                        reason = 'supervisor_effective_memory_reserve'
                    elif time.monotonic()-begin > args.seconds:
                        reason = 'supervisor_wall_time_budget'
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
                time.sleep(.05)
            code = process.wait()
        record = {'method': method, 'exit_code': code, 'supervisor_stop_reason': reason,
                  'wall_s': time.monotonic()-begin, 'peak_process_tree_rss_bytes': peak}
        profile = destination / 'profile.json'
        if profile.exists():
            record['profile'] = json.loads(profile.read_text(encoding='utf-8'))
        report['paths'].append(record)
        save(args.output / 'summary.json', report)
        print(json.dumps({'method': method, 'exit_code': code, 'stop_reason': reason,
                          'status': record.get('profile', {}).get('status')}), flush=True)
    report['production_source_unchanged'] = all(
        hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
        for name, digest in report['source_hashes'].items())
    report['status'] = 'diagnosis_finished_not_acceptance'
    save(args.output / 'summary.json', report)
    return 0 if report['production_source_unchanged'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=(*METHODS, RESEARCH_METHOD))
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, RESEARCH_METHOD), default=list(METHODS))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--memory-mib', type=float, default=8192)
    parser.add_argument('--reserve-mib', type=float, default=2048)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.memory_mib, args.reserve_mib)):
        parser.error('Budgets must be finite and positive')
    if args.threads < 1 or len(set(args.methods)) != len(args.methods):
        parser.error('Positive thread count and unique methods required')
    if args.worker and args.method is None:
        parser.error('Worker requires one fixed method')
    return worker(args) if args.worker else suite(args)


if __name__ == '__main__':
    raise SystemExit(main())
