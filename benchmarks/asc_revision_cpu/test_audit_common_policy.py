"""Synthetic audit fixtures, not performance evidence."""
import json
import pytest
from benchmarks.asc_revision_cpu.test_audit_supplement30 import suite_fixture
from benchmarks.asc_revision_cpu.audit_common_policy_pairs import audit


@pytest.fixture
def common_fixture(suite_fixture):
    path=suite_fixture/'summary.json';summary=json.loads(path.read_text())
    summary['smoke_only']=False
    for spec in summary['workloads']:spec['bridge_gate_count']=6
    path.write_text(json.dumps(summary),encoding='utf-8')
    for kind in ['chain','ring','nonlocal_mixed']:
        path=suite_fixture/kind/'result.json';data=json.loads(path.read_text())
        data.update(qubits=30,smoke_only=False,original30_complete=False,all_contest_tasks_complete=False,affinity=list(range(16)),
                    outer_threadpools=[{'num_threads':1,'user_api':'blas'}],
                    common_thread_policy='other BLAS=1; QR width<512:1,<1024:4,<2048:8,else16; shared by ALL methods')
        for key in ['full_state','means_s','stdev_s']:
            data[key]['controlled_rank2']=data[key].pop('guarded_scaled_candidate')
        data['full_state']['controlled_rank2']['preparation']={'controlled_rank2_splits':30}
        data['speedups_vs_official_common_qr_policy']={'integrated':5.,'controlled_rank2':5.}
        for row in data['rows']:
            if row['method']=='guarded_scaled_candidate':row['method']='controlled_rank2'
            row['qr_thread_call_counts']={'1':100,'4':0,'8':0,'16':0}
        path.write_text(json.dumps(data),encoding='utf-8')
    return suite_fixture


def test_valid_case_scope_not_original30(common_fixture):
    result=audit(common_fixture)
    assert result['supplement30_suite_complete']
    assert result['aggregate_speedups_vs_official_common_qr_policy']['controlled_rank2']==5.
    assert not result['original30_complete'] and not result['all_contest_tasks_complete']


@pytest.mark.parametrize('fault',['partial','false7','pool','qr_histogram','rank2_not_used','source','original_promoted'])
def test_reject_false_or_incomplete_claim(common_fixture,fault):
    path=common_fixture/'chain/result.json';data=json.loads(path.read_text())
    if fault=='partial':data['rows'].pop()
    elif fault=='false7':data['speedups_vs_official_common_qr_policy']['controlled_rank2']=7.
    elif fault=='pool':data['outer_threadpools'][0]['num_threads']=16
    elif fault=='qr_histogram':data['rows'][0]['qr_thread_call_counts']={'64':10}
    elif fault=='rank2_not_used':data['full_state']['controlled_rank2']['preparation']['controlled_rank2_splits']=0
    elif fault=='source':(common_fixture/'source_snapshot/synthetic.py').write_bytes(b'# changed')
    elif fault=='original_promoted':data['original30_complete']=True
    path.write_text(json.dumps(data),encoding='utf-8')
    result=audit(common_fixture)
    assert not result['supplement30_suite_complete']
    assert result['aggregate_speedups_vs_official_common_qr_policy'] is None
    assert not result['cases'][0]['pass']
    assert result['cases'][0]['speedups_vs_official_common_qr_policy'] is None
