"""Synthetic stopped checkpoints; never measured speed evidence."""
from copy import deepcopy
import pytest
from benchmarks.asc_revision_cpu.test_audit_common_policy import common_fixture
from benchmarks.asc_revision_cpu.test_audit_supplement30 import suite_fixture
from benchmarks.asc_revision_cpu.resume_common_policy_chain import METHODS,validate_checkpoint,plan
from benchmarks.asc_revision_cpu.resume_common_policy_chain import suite as resume_suite,digest
from benchmarks.asc_revision_cpu.resume_common_policy_chain import observe_rank2_splits
from benchmarks.asc_revision_cpu.paired_supplement30_common_policy import suite as fresh_suite
from argparse import Namespace
import json


@pytest.fixture
def checkpoint(common_fixture):
    summary=json.loads((common_fixture/'summary.json').read_text())
    data=json.loads((common_fixture/'chain/result.json').read_text())
    summary.update(status='incomplete_or_failed',cases=[{
        'kind':'chain','pass':False,'exit_code':-9,'stop_reason':'wall_time_budget'}])
    data.update(gates=9693,oracle_gates_applied=9693,warmup_repeats=1,measured_repeats=3,
                formal_supplementary_30q_complete=False,status='timing_integrated_0')
    for state in data['full_state'].values():state['gates_applied']=9693
    data['rows']=data['rows'][:1]
    data['rows'][0]['abs_error']=0.
    return summary,data


def test_prefix_and_unmodified_original(checkpoint):
    summary,data=checkpoint;old=deepcopy((summary,data))
    assert validate_checkpoint(summary,data)==plan()[1:]
    assert (summary,data)==old


@pytest.mark.parametrize('fault',['active','oom','exit','scope','empty','order','state','scalar','policy','precision','hist','source'])
def test_reject_invalid_continuation(checkpoint,fault):
    summary,data=checkpoint
    if fault=='active':summary['status']='running'
    elif fault=='oom':summary['cases'][0]['stop_reason']='process_tree_memory_budget'
    elif fault=='exit':summary['cases'][0]['exit_code']=0
    elif fault=='scope':data['original30_complete']=True
    elif fault=='empty':data['rows']=[]
    elif fault=='order':data['rows'][0]['method']='integrated'
    elif fault=='state':data['full_state']['controlled_rank2']['overlap_real']=.9
    elif fault=='scalar':data['rows'][0]['value']=1.
    elif fault=='policy':data['common_thread_policy']='different'
    elif fault=='precision':data['precision']='complex64'
    elif fault=='hist':data['rows'][0]['qr_thread_call_counts']={'64':100}
    elif fault=='source':summary['source_unchanged']=False
    with pytest.raises(ValueError):validate_checkpoint(summary,data)


def test_full_smoke_pipeline_preserves_prior_bytes_and_rows(tmp_path):
    # Intentionally synthetic 9q stop; never used as experimental evidence.
    source=tmp_path/'synthetic_stopped_smoke';output=tmp_path/'synthetic_resumed_smoke'
    fresh_suite(Namespace(output=source,execute=True,smoke=True,seconds=120.,memory_mib=512.))
    summary_path=source/'summary.json';summary=json.loads(summary_path.read_text())
    summary.update(status='incomplete_or_failed',synthetic_fixture_not_measurement=True,
                   cases=[{'kind':'chain','pass':False,'exit_code':-9,'stop_reason':'wall_time_budget'}])
    summary_path.write_text(json.dumps(summary),encoding='utf-8')
    result_path=source/'chain/result.json';data=json.loads(result_path.read_text())
    data.update(status='synthetic_stopped_fixture',formal_supplementary_30q_complete=False,rows=data['rows'][:4])
    result_path.write_text(json.dumps(data),encoding='utf-8')
    hashes={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    resume_suite(Namespace(source=source,output=output,execute=True,smoke=True,seconds=120.,memory_mib=512.))
    resumed=json.loads((output/'chain/result.json').read_text())
    final=json.loads((output/'summary.json').read_text())
    assert resumed['status']=='pass_smoke_only' and len(resumed['rows'])==12
    assert resumed['rows'][:4]==data['rows']
    assert [r['method'] for r in resumed['restart_warmups']]==list(METHODS)
    assert all(r['repeat']==-1 for r in resumed['restart_warmups'])
    assert not resumed['formal_supplementary_30q_complete'] and not final['supplement30_suite_complete']
    assert resumed['full_state']==data['full_state']
    assert all(digest(source/name)==sha==digest(output/'prior_segment'/name) for name,sha in hashes.items())


def test_live_counter_observer_and_exception_restore():
    import numpy as np
    from qibo import gates
    from qibo.backends import NumpyBackend
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS
    original=ControlledRank2MPS._two_site
    arrays=[np.ones(shape,dtype=np.complex128) for shape in [(1,2,2),(2,1,2),(1,2,2),(2,1,2)]]
    state=ControlledRank2MPS(arrays,shape='lrp')
    with observe_rank2_splits() as evidence:
        state._two_site(gates.CNOT(1,2).matrix(NumpyBackend()),1,2,'right')
    assert evidence['rank2_splits_observed']==1 and ControlledRank2MPS._two_site is original
    with pytest.raises(RuntimeError):
        with observe_rank2_splits():raise RuntimeError('synthetic')
    assert ControlledRank2MPS._two_site is original


def test_reject_recursive_backup_destination(tmp_path):
    with pytest.raises(ValueError,match='inside the preserved source'):
        resume_suite(Namespace(source=tmp_path,output=tmp_path/'child',execute=False))
