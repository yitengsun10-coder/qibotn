"""Strict audit of completed hardware-limited deep pressure evidence."""
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.asc_revision_cpu.audit_upstream_mixed import validate_result

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('upstream_typo_only', 'integrated')


def validate_continuation(data, summary, original, original_summary):
    from benchmarks.asc_revision_cpu.resume_pressure_chain import validate_checkpoint, plan
    remaining, exact = validate_checkpoint(original, original_summary)
    continuation = data['continuation']
    count = len(original['rows'])
    if (data['rows'][:count] != original['rows'] or data['full_state'] != original['full_state'] or
            data['source_hashes'] != original['source_hashes'] or data['reference'] != original['reference']):
        raise ValueError('Original completed evidence was modified during continuation')
    if (continuation['preserved_row_count'] != count or continuation['remaining_plan'] != [list(k) for k in remaining] or
            continuation['original_exit_code'] != original_summary['exit_code'] or
            continuation['original_stop_reason'] != original_summary['stop_reason'] or
            summary.get('segmented_run_disclosed') is not True or
            summary.get('original_segment_exit_code') != original_summary['exit_code'] or
            summary.get('original_segment_stop_reason') != original_summary['stop_reason']):
        raise ValueError('Restart/stop provenance is missing or inconsistent')
    if [(row['method'], row['repeat']) for row in data['rows']] != plan():
        raise ValueError('Continuation changed the declared measurement order')
    warmups = data.get('restart_warmups', [])
    if [(row['method'], row['repeat']) for row in warmups] != [(method, -1) for method in METHODS]:
        raise ValueError('Missing fresh-process warmup for either backend')
    for row in warmups + data['rows'][count:]:
        if (row.get('segment') != 'resumed_process' or row['exact_value'] != exact or
                not all(math.isfinite(row[key]) for key in ('runtime_s', 'value', 'abs_error')) or
                row['runtime_s'] <= 0 or row['abs_error'] > 1e-10 or abs(row['value']-exact) > 1e-10):
            raise ValueError('Invalid resumed call or independent reference')
    if not data.get('resumed_threadpools') or any(pool['num_threads'] != 1 for pool in data['resumed_threadpools']):
        raise ValueError('Resumed actual thread count differs')
    return {'segmented_run': True, 'preserved_rows': count, 'resumed_rows': len(remaining),
            'additional_restart_warmups': len(warmups), 'original_stop_reason': original_summary['stop_reason'],
            'original_exit_code': original_summary['exit_code']}


def validate_pressure(data, summary):
    if data.get('calibration_only') is not False or summary.get('calibration_only') is not False:
        raise ValueError('Calibration or unknown provenance is not acceptance')
    if data.get('qubits') != 20 or data.get('layers') != 169 or data.get('gates') != 9991 or data.get('depth') != 677:
        raise ValueError('Pressure input does not match the predeclared 20q workload')
    report = validate_result(data, summary, METHODS)
    if set(data['full_state']) != set(METHODS):
        raise ValueError('Both official and optimized full states must be independently checked')
    state = data['full_state']['upstream_typo_only']
    if (not all(math.isfinite(state[k]) for k in ('norm_squared', 'fidelity', 'max_state_error')) or
            abs(state['norm_squared'] - 1) > 1e-10 or abs(state['fidelity'] - 1) > 1e-10 or
            state['max_state_error'] > 1e-9):
        raise ValueError('Official full-state correctness failed')
    if not data.get('threadpools') or any(pool['num_threads'] != 1 for pool in data['threadpools']):
        raise ValueError('Actual BLAS thread pools are not the declared matched one-thread setting')
    ratio = report['speedups_vs_upstream_typo_only']['integrated']
    if not math.isclose(data['speedup'], ratio, rel_tol=1e-12):
        raise ValueError('Recorded speedup differs from original timing rows')
    return {**report, 'full_state_methods_verified': 2, 'speedup': ratio}


def audit(directory, require_current=True):
    data = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    report = validate_pressure(data, summary)
    if 'continuation' in data:
        original_path, summary_path = directory/'prior_segment/result.json', directory/'prior_segment/summary.json'
        original = json.loads(original_path.read_text(encoding='utf-8'))
        original_summary = json.loads(summary_path.read_text(encoding='utf-8'))
        continuation = data['continuation']
        for path, key in ((original_path, 'original_result_sha256'), (summary_path, 'original_summary_sha256'),
                          (directory/'resume_pressure_chain_snapshot.py', 'resume_script_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != continuation[key]:
                raise ValueError('Continuation evidence hash mismatch: ' + str(path))
        report.update(validate_continuation(data, summary, original, original_summary))
    qasm = (directory / 'input.qasm').read_text(encoding='utf-8')
    if hashlib.sha256(qasm.encode()).hexdigest() != data['qasm_sha256']:
        raise ValueError('QASM hash mismatch')
    for relative, digest in data['source_hashes'].items():
        path = ROOT / relative
        snapshot = directory / (path.stem + '_snapshot.py')
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError('Measured source snapshot mismatch: ' + relative)
        if require_current and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Current source differs from measured source: ' + relative)
    from benchmarks.asc_revision_cpu.upstream_reference import source_pair, UPSTREAM_COMMIT
    original = (directory / 'upstream_quimb_original.py').read_text(encoding='utf-8')
    repaired = (directory / 'upstream_quimb_typo_only.py').read_text(encoding='utf-8')
    if data['reference']['commit'] != UPSTREAM_COMMIT or (original, repaired) != source_pair():
        raise ValueError('Official reference source/sole typo repair mismatch')
    for key, text in (('original_sha256', original), ('typo_only_sha256', repaired)):
        if hashlib.sha256(text.encode()).hexdigest() != data['reference'][key]:
            raise ValueError('Official reference hash mismatch')
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit
    circuit = build_circuit(Workload(data['qubits'], data['layers']))
    if (circuit.to_qasm() != qasm or len(circuit.queue) != data['gates'] or circuit.depth != data['depth'] or
            dict(Counter(type(g).__name__ for g in circuit.queue)) != data['gate_counts']):
        raise ValueError('Predeclared input cannot be reconstructed')
    return {'recorded_at': datetime.now().astimezone().isoformat(), 'status': 'verified',
        'directory': str(directory.resolve()), **report,
        'qubits': data['qubits'], 'gates': data['gates'], 'depth': data['depth'],
        'source_snapshots_verified': len(data['source_hashes']),
        'current_source_matches_required': require_current,
        'hardware_limited_deep20_pressure_complete': True,
        'formal_30q_acceptance_complete': False,
        'seven_x_on_this_input': report['speedup'] >= 7,
        'all_requirements_complete': False,
        'scope': '20q reduced-qubit but full-depth/gate-count CPU pressure evidence; not a 30q or universal speed claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    report = audit(args.directory, require_current=not args.snapshot_only)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
