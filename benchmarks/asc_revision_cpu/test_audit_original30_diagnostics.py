"""Synthetic audit fixtures only: no quantum performance/acceptance evidence."""
import hashlib
import json

import pytest
from benchmarks.asc_revision_cpu import audit_original30_diagnostics as module


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    data=b'synthetic unit-test input; never performance evidence\n'
    digest=hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(module,'QASM_SHA256',digest)
    source=b'synthetic source snapshot\n'
    (tmp_path/'input.qasm').write_bytes(data)
    target=tmp_path/'source_snapshot/src/example.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    summary={'input':{'kind':'original30','input_file':'input.qasm','qubits':30,'gates':9998,
        'depth':449,'input_sha256':digest},'threads':16,'candidate':'short',
        'formal_30q_acceptance_complete':False,'publishable_speedup':None,
        'source_unchanged':True,'source_hashes':{'src/example.py':hashlib.sha256(source).hexdigest()},
        'status':'bounded_or_failed','exit_code':-9,'supervisor_stop_reason':'supervisor_wall_time_budget',
        'wall_s':180.2,'peak_process_tree_rss_bytes':500000000}
    profile={'input_sha256':digest,'qubits':30,'gates':9998,'depth':449,'precision':'complex128',
        'cutoff':0,'max_bond':None,'formal_30q_acceptance_complete':False,'publishable_speedup':None,
        'affinity':list(range(160,176)),'max_bond_observed':4096,'completed_apply_gate_calls':729,
        'status':'running'}
    return tmp_path,summary,profile


def put(fixture):
    path,summary,profile=fixture
    (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (path/'profile.json').write_text(json.dumps(profile),encoding='utf-8')
    return path


def test_incomplete_receipt_never_promotes_to_speed_or_acceptance(fixture):
    report=module.audit([put(fixture)])
    assert report['status']=='verified_diagnostic_evidence_only'
    assert report['publishable_speedup'] is None
    assert report['independent_original30_full_state_verified'] is False
    assert report['formal_30q_acceptance_complete'] is False


@pytest.mark.parametrize('mutation',['input','source','scale','precision','cutoff','formal',
    'speed','running','source_freeze','affinity','unsafe_source','missing_source'])
def test_reject_inconsistent_or_falsely_promoted_evidence(fixture,mutation):
    path,summary,profile=fixture
    if mutation=='input': (path/'input.qasm').write_bytes(b'changed')
    elif mutation=='source': (path/'source_snapshot/src/example.py').write_bytes(b'changed')
    elif mutation=='scale': summary['input']['qubits']=20
    elif mutation=='precision': profile['precision']='complex64'
    elif mutation=='cutoff': profile['cutoff']=1e-10
    elif mutation=='formal': summary['formal_30q_acceptance_complete']=True
    elif mutation=='speed': profile['publishable_speedup']=7
    elif mutation=='running': summary['status']='running'
    elif mutation=='source_freeze': summary['source_unchanged']=False
    elif mutation=='affinity': profile['affinity']=[0]*16
    elif mutation=='unsafe_source': summary['source_hashes']={'../outside.py':'bad'}
    elif mutation=='missing_source': summary['source_hashes']={}
    with pytest.raises(ValueError): module.audit([put(fixture)])
