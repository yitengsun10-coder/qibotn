"""Synthetic continuation audits, not experimental performance evidence."""
from copy import deepcopy
import hashlib
import json
import shutil
import pytest
from benchmarks.asc_revision_cpu.test_audit_common_policy import common_fixture
from benchmarks.asc_revision_cpu.test_audit_supplement30 import suite_fixture
from benchmarks.asc_revision_cpu.audit_resumed_common_policy_chain import audit
from benchmarks.asc_revision_cpu.resume_common_policy_chain import METHODS,SOURCES,ROOT,plan,digest


def save(path,data):
    path.write_text(json.dumps(data),encoding='utf-8')


@pytest.fixture
def resumed_fixture(common_fixture,tmp_path):
    source=common_fixture;summary=json.loads((source/'summary.json').read_text())
    data=json.loads((source/'chain/result.json').read_text())
    encoded=json.dumps({'qubits':30,'gates':[{'name':'fixture_only'}]*9693}).encode()
    (source/'chain/input.json').write_bytes(encoded);sha=hashlib.sha256(encoded).hexdigest()
    summary['workloads'][0].update(gates=9693,input_sha256=sha)
    data.update(gates=9693,input_sha256=sha,oracle_gates_applied=9693,warmup_repeats=1,measured_repeats=3)
    for state in data['full_state'].values():state['gates_applied']=9693
    # Real frozen preparation loses private counters when Circuit.psi copies.
    data['full_state']['controlled_rank2']['preparation']['controlled_rank2_splits']=0
    for row in data['rows']:row['abs_error']=0.
    all_rows=deepcopy(data['rows'])
    data.update(formal_supplementary_30q_complete=False,status='timing_integrated_1',rows=data['rows'][:4])
    summary.update(status='incomplete_or_failed',source_hashes={},environment={'versions':{'numpy':'fixture'}},
                   cases=[{'kind':'chain','exit_code':-9,'stop_reason':'wall_time_budget','pass':False}])
    for path in SOURCES:
        name=path.relative_to(ROOT).as_posix();target=source/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        summary['source_hashes'][name]=digest(path)
    save(source/'summary.json',summary);save(source/'chain/result.json',data)
    output=tmp_path.parent/(tmp_path.name+'_resumed');output.mkdir()
    shutil.copytree(source,output/'prior_segment');shutil.copytree(source/'source_snapshot',output/'source_snapshot')
    inventory={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    script='benchmarks/asc_revision_cpu/resume_common_policy_chain.py'
    shutil.copyfile(ROOT/script,output/'source_snapshot'/script)
    shutil.copytree(source/'chain',output/'chain')
    final=deepcopy(data)
    final.update(status='pass',formal_supplementary_30q_complete=True,rows=all_rows,
                 resumed_affinity=data['affinity'],resumed_versions={'numpy':'fixture'},
                 resumed_outer_threadpools=[{'user_api':'blas','num_threads':1}],
                 restart_warmups=[{**deepcopy(all_rows[i]),'repeat':-1} for i in range(3)],
                 continuation={'prior_files_sha256':inventory,'preserved_row_count':4,
                    'remaining_plan':[list(x) for x in plan()[4:]],'resume_script_sha256':digest(ROOT/script),
                    'original_exit_code':-9,'original_stop_reason':'wall_time_budget'})
    final['restart_warmups'][2].update(rank2_splits_observed=30,trace_scope='unmeasured_restart_warmup_only')
    save(output/'chain/result.json',final)
    report=deepcopy(summary)
    report['source_hashes'][script]=digest(ROOT/script)
    report.update(status='complete_chain_only',chain_continuation_complete=True,segmented_run_disclosed=True,
                  supplement30_suite_complete=False,
                  cases=[{'kind':'chain','exit_code':0,'stop_reason':None,'pass':True,'peak_process_tree_rss_bytes':1000}])
    save(output/'summary.json',report)
    return output


def test_complete_chain_does_not_promote_full_suite(resumed_fixture):
    result=audit(resumed_fixture)
    assert result['formal_chain_supplement_complete']
    assert not result['supplement30_suite_complete'] and not result['original30_complete']
    assert result['chain']['speedups_vs_official_common_qr_policy']['controlled_rank2']==5.


@pytest.mark.parametrize('fault',['prior','prefix','warmup','version','affinity','false7','scope','partial','counter','trace_scope'])
def test_reject_changed_or_incomplete_resume(resumed_fixture,fault):
    path=resumed_fixture/'chain/result.json';data=json.loads(path.read_text())
    if fault=='prior':(resumed_fixture/'prior_segment/chain/run.log').write_bytes(b'changed')
    elif fault=='prefix':data['rows'][0]['runtime_s']+=1.
    elif fault=='warmup':data['restart_warmups'].pop()
    elif fault=='version':data['resumed_versions']['numpy']='changed'
    elif fault=='affinity':data['resumed_affinity']=[0]
    elif fault=='false7':data['speedups_vs_official_common_qr_policy']['controlled_rank2']=7.
    elif fault=='scope':data['original30_complete']=True
    elif fault=='partial':data['rows'].pop()
    elif fault=='counter':data['restart_warmups'][2]['rank2_splits_observed']=0
    elif fault=='trace_scope':data['restart_warmups'][2]['trace_scope']='formal_timing_changed'
    save(path,data)
    result=audit(resumed_fixture)
    assert not result['formal_chain_supplement_complete'] and result['chain'] is None
