"""Verify bounded original30 evidence without promoting it to acceptance.

This is transport/source/input/status auditing, NOT an independent quantum
state check or a performance audit. Partial calls never produce a speed ratio.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from benchmarks.asc_revision_cpu.profile_fixed30_bounded import QASM_SHA256


def audit_one(folder):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    profile = json.loads((folder / 'profile.json').read_text(encoding='utf-8'))
    spec = summary['input']
    if (spec['qubits'], spec['gates'], spec['depth'], spec['input_sha256']) != (30,9998,449,QASM_SHA256):
        raise ValueError('Original30 fixed specification changed')
    if spec['kind'] != 'original30' or spec['input_file'] != 'input.qasm':
        raise ValueError('Wrong diagnostic input declaration')
    if hashlib.sha256((folder / 'input.qasm').read_bytes()).hexdigest() != QASM_SHA256:
        raise ValueError('Raw original30 input changed')
    for record in (summary, profile):
        if record.get('formal_30q_acceptance_complete') is not False or record.get('publishable_speedup') is not None:
            raise ValueError('Diagnostic falsely claims acceptance or speed')
    if not summary.get('source_unchanged'):
        raise ValueError('Runtime source freeze failed')
    if not summary['source_hashes']:
        raise ValueError('Missing source snapshots')
    for name, digest in summary['source_hashes'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe source declaration')
        if hashlib.sha256((folder / 'source_snapshot' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source snapshot differs')
    if (profile['input_sha256'], profile['precision'], profile['cutoff'], profile['max_bond']) != (QASM_SHA256,'complex128',0,None):
        raise ValueError('Profile input or precision changed')
    if (profile['qubits'], profile['gates'], profile['depth']) != (30,9998,449):
        raise ValueError('Profile scale changed')
    affinity = profile['affinity']
    if len(affinity) != 16 or len(set(affinity)) != 16 or summary['threads'] != 16:
        raise ValueError('Common 16-CPU affinity required')
    if summary['status'] == 'bounded_or_failed':
        if summary['exit_code'] != -9 or summary['supervisor_stop_reason'] != 'supervisor_wall_time_budget':
            raise ValueError('Unexpected failure requires a separate investigation')
        status = 'time_guard_incomplete_original30'
    elif summary['status'] == 'complete_diagnostic_only':
        if summary['exit_code'] != 0 or summary['supervisor_stop_reason'] is not None:
            raise ValueError('Inconsistent diagnostic completion')
        if profile['status'] != 'completed_instrumented_call_not_acceptance':
            raise ValueError('Incomplete worker cannot claim completed call')
        status = 'completed_instrumented_call_only_not_acceptance'
    else:
        raise ValueError('Running/plan evidence is not a finished diagnostic')
    return {'folder': str(folder), 'status': status, 'candidate': summary['candidate'],
        'wall_s': summary['wall_s'], 'peak_process_tree_rss_bytes': summary['peak_process_tree_rss_bytes'],
        'source_snapshot_count': len(summary['source_hashes']), 'affinity': affinity,
        'max_bond_observed': profile['max_bond_observed'],
        'api_gate_calls_not_original_gates': profile['completed_apply_gate_calls'],
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None}


def audit(folders):
    rows = [audit_one(folder.resolve()) for folder in folders]
    if not rows or any(row['affinity'] != rows[0]['affinity'] for row in rows):
        raise ValueError('All diagnostics require the same nonempty affinity')
    return {'status': 'verified_diagnostic_evidence_only', 'diagnostics': rows,
        'independent_original30_full_state_verified': False,
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None,
        'note': 'No partial progress ratio, no original30 correctness or performance acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.folder)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
