"""Synthetic audit fixtures only; no fixture is an experimental result."""
from copy import deepcopy
import hashlib
import json
import pytest
from benchmarks.asc_revision_cpu.audit_supplement30 import audit,METHODS,KINDS
from benchmarks.asc_revision_cpu.upstream_reference import source_pair,UPSTREAM_COMMIT


def save(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def suite_fixture(tmp_path):
    original,repaired=(s.encode('utf-8') for s in source_pair())
    reference={'commit':UPSTREAM_COMMIT,'original_sha256':hashlib.sha256(original).hexdigest(),
               'typo_only_sha256':hashlib.sha256(repaired).hexdigest(),'extra_norm_contraction':False,
               'state_preparations_per_call':1,'algorithm_or_backend_replacement':False}
    snapshot=tmp_path/'source_snapshot';snapshot.mkdir()
    (snapshot/'synthetic.py').write_bytes(b'# Synthetic audit unit fixture, not a run\n')
    summary={'status':'pass_supplementary_only','source_unchanged':True,
             'original30_complete':False,'all_contest_tasks_complete':False,
             'source_hashes':{'synthetic.py':hashlib.sha256((snapshot/'synthetic.py').read_bytes()).hexdigest()},
             'budgets_per_case':{'rss_mib':4096},'cases':[],'workloads':[]}
    for kind in KINDS:
        path=tmp_path/kind;path.mkdir()
        encoded=json.dumps({'qubits':30,'gates':[{'name':'fixture_only'}]*9700}).encode('utf-8')
        (path/'input.json').write_bytes(encoded)
        (path/'upstream_quimb_original.py').write_bytes(original)
        (path/'upstream_quimb_typo_only.py').write_bytes(repaired)
        spec={'kind':kind,'gates':9700,'input_sha256':hashlib.sha256(encoded).hexdigest()}
        data={**spec,'status':'pass','formal_supplementary_30q_complete':True,'reference':reference,
              'depth':450,'cutoff':0,'max_bond':None,'precision':'complex128','oracle_gates_applied':9700,
              'independent_observable':.25,'threads':8,'threadpools':[{'num_threads':8,'user_api':'blas'}],
              'baseline_thread_calibration':[{'threads':n,'runtime_s':t,'value':.25} for n,t in [(1,3.),(4,2.),(8,1.)]],
              'full_state':{m:{'oracle_norm_squared':1.,'mps_norm_squared':1.,'overlap_real':1.,'overlap_imag':0.,
                               'full_state_comparison_pass':True,'gates_applied':9700,'oracle_branches':64} for m in METHODS},
              'rows':[{'method':m,'repeat':r,'runtime_s':10. if m==METHODS[0] else 2.,'value':.25}
                      for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])],
              'means_s':{m:10. if m==METHODS[0] else 2. for m in METHODS},
              'stdev_s':{m:0. for m in METHODS},'speedups_vs_tuned_official':{m:5. for m in METHODS[1:]}}
        save(path/'result.json',data)
        summary['workloads'].append(spec)
        summary['cases'].append({'kind':kind,'exit_code':0,'stop_reason':None,'pass':True,'peak_process_tree_rss_bytes':1000})
    save(tmp_path/'summary.json',summary)
    return tmp_path


def test_valid_synthetic_fixture_has_correct_scope(suite_fixture):
    result=audit(suite_fixture)
    assert result['supplement30_suite_complete']
    assert result['aggregate_speedups']['integrated']==5.
    assert not result['original30_complete'] and not result['all_contest_tasks_complete']


@pytest.mark.parametrize('fault',['partial','false7x','extra_norm','changed_input','bad_state','zero_norm',
                                'missing_threads','wrong_threads','baseline_source','snapshot','external_stop'])
def test_reject_partial_or_false_claims(suite_fixture,fault):
    path=suite_fixture/'chain/result.json'
    data=json.loads(path.read_text())
    if fault=='partial':data['rows'].pop()
    elif fault=='false7x':data['speedups_vs_tuned_official']['integrated']=7.
    elif fault=='extra_norm':data['reference']['extra_norm_contraction']=True
    elif fault=='changed_input':(suite_fixture/'chain/input.json').write_bytes(b'{}')
    elif fault=='bad_state':data['full_state']['integrated']['overlap_real']=.9
    elif fault=='zero_norm':data['full_state']['integrated']['mps_norm_squared']=0.
    elif fault=='missing_threads':data['threadpools']=[]
    elif fault=='wrong_threads':data['threads']=4
    elif fault=='baseline_source':(suite_fixture/'chain/upstream_quimb_typo_only.py').write_bytes(b'# changed')
    elif fault=='snapshot':(suite_fixture/'source_snapshot/synthetic.py').write_bytes(b'# changed')
    elif fault=='external_stop':(suite_fixture/'external_drain_record.json').write_bytes(b'{}')
    save(path,data)
    result=audit(suite_fixture)
    assert not result['supplement30_suite_complete']
    assert result['aggregate_speedups'] is None
    assert result['failures']
