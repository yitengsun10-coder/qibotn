"""Bounded original30 diagnosis with identical size-only QR thread policy.

Research variants are compared as diagnostics, not formal speed measurements.
Incomplete progress counters never establish a speedup or exact30 acceptance.
The existing supervisor preserves source/input snapshots, process-tree memory,
cgroup reserve and wall-time protections. No cloud instance is created here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from benchmarks.asc_revision_cpu import probe_controlled_rank2 as previous
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import clone
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import (
    ROOT, Recorder, ProbeStop, effective_memory, save,
)

HERE = Path(__file__).resolve().parent
SOURCES = list(dict.fromkeys([*previous.SOURCES, Path(__file__),
    HERE / 'candidate_short_cnot_rank2.py', HERE / 'common_qr_thread_policy.py']))
suite = clone(previous.suite, {'SOURCES': SOURCES, '__file__': __file__})


def worker(args):
    import numpy as np
    import psutil
    import quimb.tensor as qtn
    from qibo.backends import NumpyBackend
    from threadpoolctl import threadpool_info, threadpool_limits
    from qibotn.backends import _exact_numpy as exact
    from benchmarks.asc_large_cpu.benchmark import make_observable
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS
    from benchmarks.asc_revision_cpu.common_qr_thread_policy import common_qr_policy

    if args.candidate == 'short':
        from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import make_backend
    else:
        from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import make_backend
    declaration = json.loads((args.output / 'summary.json').read_text(encoding='utf-8'))
    for name, digest in declaration['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed after predeclaration')
    process = psutil.Process()
    allowed = process.cpu_affinity()
    if args.threads > len(allowed):
        raise ValueError('Affinity exceeds allowed CPUs')
    process.cpu_affinity(allowed[:args.threads])
    circuit, encoded, spec = previous.build_input(args.kind, args.smoke)
    if spec != declaration['input']:
        raise ValueError('Predeclared fixed input changed')
    (args.output / spec['input_file']).write_bytes(encoded)
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    labels, sites, coefficients = make_observable(circuit.nqubits)
    result = {**spec, 'status': 'running', 'candidate': args.candidate,
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None,
        'instrumented_not_formal_timing': True, 'production_integrated': False,
        'precision': 'complex128', 'cutoff': 0, 'max_bond': None,
        'smoke_test_only': args.smoke, 'affinity': process.cpu_affinity(),
        'thread_policy': 'all other BLAS1; QR width-only common policy 1/4/8/16',
        'max_bond_observed': 1, 'controlled_rank2_splits': 0,
        'completed_apply_gate_calls': 0,
        'counter_note': 'API calls, not original gates; no partial-path speed ratio'}
    begin = time.perf_counter()
    last_write, last_state = 0., None
    def guard(extra):
        if process.memory_info().rss + extra > args.memory_mib * 2**20:
            raise ProbeStop('process_rss_or_projected_workspace_budget')
        if effective_memory(psutil.virtual_memory().available)['available_bytes'] - extra < args.reserve_mib * 2**20:
            raise ProbeStop('effective_memory_or_projected_workspace_reserve')
        if time.perf_counter() - begin > args.seconds:
            raise ProbeStop('worker_time_budget')
    def observe(force=False):
        nonlocal last_write
        now = time.perf_counter()
        if not force and now - last_write < 1.:
            return
        result.update(instrumented_elapsed_s=now - begin, stages=recorder.stages,
            in_flight_stages=[frame[0] for frame in recorder.stack],
            memory=effective_memory(psutil.virtual_memory().available),
            current_process_rss_bytes=process.memory_info().rss)
        save(args.output / 'profile.json', result)
        last_write = now
    def after_gate(arguments, keywords):
        nonlocal last_state
        last_state = arguments[0]._psi
        result['max_bond_observed'] = max(result['max_bond_observed'], last_state.max_bond())
        result['controlled_rank2_splits'] = getattr(last_state, 'controlled_rank2_splits', 0)
        result['completed_apply_gate_calls'] += 1
    def matrix_metadata(arguments, keywords):
        matrix = arguments[0]
        return {'input_bytes': matrix.nbytes, 'shape': list(matrix.shape),
                'estimated_extra_bytes': 4 * matrix.nbytes}
    recorder = Recorder(guard, observe)
    with threadpool_limits(limits=1), common_qr_policy() as policy_calls:
        result['other_blas_threadpools'] = threadpool_info()
        if any(pool['num_threads'] != 1 for pool in result['other_blas_threadpools']
               if pool['user_api'] == 'blas'):
            raise ValueError('Other BLAS1 limit ineffective')
        result['qr_policy_calls'] = policy_calls
        try:
            recorder.patch(qtn.CircuitMPS, 'apply_gate', 'apply_gate_inclusive', after=after_gate)
            recorder.patch(qtn.CircuitMPS, 'apply_gate_raw', 'apply_gate_raw_inclusive', after=after_gate)
            recorder.patch(ControlledRank2MPS, '_two_site', 'controlled_rank2.two_site')
            recorder.patch(exact, '_left_isometric_factor', 'exact.isometric_factor', before=matrix_metadata)
            recorder.patch(exact, '_reduced_qr', 'exact.reduced_qr', before=matrix_metadata)
            observe(True)
            value = float(backend.exp_value_observable_symbolic(
                circuit, labels, sites, coefficients, circuit.nqubits))
            if not math.isfinite(value):
                raise ValueError('Nonfinite observable')
            result.update(status='completed_instrumented_call_not_acceptance',
                          observable=value, instrumented_call_s=time.perf_counter() - begin)
        except ProbeStop as error:
            result.update(status='bounded_incomplete', stop_reason=str(error))
        finally:
            recorder.restore()
            observe(True)
        if args.smoke and result['status'] == 'completed_instrumented_call_not_acceptance':
            oracle = NumpyBackend().execute_circuit(circuit).state()
            actual = np.asarray(last_state.to_dense()).reshape(-1)
            result['smoke_max_abs_error'] = float(np.max(np.abs(actual - oracle)))
            if result['smoke_max_abs_error'] > 1e-10:
                raise ValueError('Independent dense smoke state mismatch')
            observe(True)
    return 0 if result['status'] == 'completed_instrumented_call_not_acceptance' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['original30'], default='original30')
    parser.add_argument('--candidate', choices=['plain', 'short'], required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--memory-mib', type=float, default=32768)
    parser.add_argument('--reserve-mib', type=float, default=2048)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.memory_mib, args.reserve_mib)):
        parser.error('Finite positive resource budgets required')
    if args.threads != 16 and not (args.smoke and 1 <= args.threads <= 16):
        parser.error('Original30 diagnosis requires common 16-CPU affinity')
    raise SystemExit(worker(args) if args.worker else suite(args))
