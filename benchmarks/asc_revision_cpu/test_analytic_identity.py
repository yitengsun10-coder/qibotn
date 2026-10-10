import gc
import weakref

import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit,gates
from qibo.backends import NumpyBackend
from qibotn.backends import _exact_numpy as exact
from benchmarks.asc_large_cpu.benchmark import Workload,build_circuit
from benchmarks.asc_revision_cpu.candidate_analytic_identity import (
    AnalyticIdentityRank2MPS,exact_identity_isometry,make_backend,prepare)
from benchmarks.asc_revision_cpu.evaluate import build_circuit as build_general


def identity_layout(width,side,dtype=complex):
    matrix=np.eye(2*width,dtype=dtype)
    if side=='left':return matrix.reshape(width,2,2*width).transpose(0,2,1)
    return matrix.reshape(2*width,width,2)


@pytest.mark.parametrize('width',[1,2,7,32])
@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('dtype',[np.float64,np.complex128,np.int64])
def test_exact_guard_scans_all_values_in_bounded_views(width,side,dtype):
    array=identity_layout(width,side,dtype)
    chunks=[]
    assert exact_identity_isometry(array,side,64,chunks.append)
    assert chunks and max(chunks)<=64
    original=array.copy()
    array[0,0,0]=.5 if dtype!=np.int64 else 2
    assert not exact_identity_isometry(array,side,64)
    array[:]=original
    # An off-diagonal edit is rejected even when very small, no tolerance.
    array[0,0,1]=1e-300 if dtype!=np.int64 else 1
    assert not exact_identity_isometry(array,side,64)


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('value',[np.nan,np.inf,1+1e-300j])
def test_nonfinite_or_nonexact_entries_rejected(side,value):
    array=identity_layout(3,side)
    array[0,0,0]=value
    assert not exact_identity_isometry(array,side,96)


@pytest.mark.parametrize('array,side',[(np.eye(3),'left'),(np.ones((2,3,2)),'left'),
    (np.ones((3,2,2)),'right'),(np.ones((2,4,3)),'left'),
    (np.ones((2,4,2),dtype=object),'left'),(np.ones((2,4,2)),'unknown')])
def test_wrong_shape_dtype_or_side_fails_closed(array,side):
    assert not exact_identity_isometry(array,side)


def tagged_state(side):
    state=AnalyticIdentityRank2MPS(qtn.MPS_computational_state('000'))
    site=0 if side=='left' else 2
    state._put(site,state._layout(site),identity_layout(1,side),side)
    return state,site


@pytest.mark.parametrize('side',['left','right'])
def test_weak_tag_has_no_snapshot_and_rejects_external_mutation(side):
    state,site=tagged_state(side)
    reference,tag=state._identity_gauges[site]
    assert isinstance(reference,weakref.ReferenceType) and tag==side
    assert state._known_identity(site)==side
    assert state.identity_guard_stats['dense_guard_snapshot_allocations']==0
    assert state.identity_guard_stats['max_tagged_array_bytes']==state[site].data.nbytes
    state[site].data.flat[0]=.5
    assert state._known_identity(site) is None
    assert state.identity_guard_stats['rejected_checks']==1


@pytest.mark.parametrize('side',['left','right'])
def test_replaced_array_not_kept_alive_by_weak_tag(side):
    state,site=tagged_state(side)
    old=weakref.ref(state[site].data)
    state[site].modify(data=state[site].data.copy())
    gc.collect()
    assert old() is None
    assert state._known_identity(site) is None


@pytest.mark.parametrize('short',[False,True])
@pytest.mark.parametrize('n,layers',[(6,4),(8,5),(10,8),(12,8),(14,12)])
@pytest.mark.parametrize('threshold',[0,256*1024])
def test_scaled_original_complete_state_and_original_output(monkeypatch,short,n,layers,threshold):
    from benchmarks.asc_large_cpu.benchmark import make_observable
    from benchmarks.asc_revision_cpu.evaluate import dense_expectation
    monkeypatch.setattr(AnalyticIdentityRank2MPS,'minimum_merged_bytes',threshold)
    observed=[]
    put=AnalyticIdentityRank2MPS._put
    def record(self,*args,**kwargs):
        put(self,*args,**kwargs)
        observed.append(dict(self.identity_guard_stats))
    monkeypatch.setattr(AnalyticIdentityRank2MPS,'_put',record)
    circuit=build_circuit(Workload(n,layers));qasm=circuit.to_qasm()
    oracle=NumpyBackend().execute_circuit(circuit).state()
    backend=make_backend(short)
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,details=prepare(backend,circuit,short)
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=6e-11)
    labels,sites,coefficients=make_observable(n)
    expected=dense_expectation(oracle,n,labels,sites,coefficients)
    actual=backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,n)
    assert actual==pytest.approx(expected,abs=6e-11)
    assert details['production_integrated'] is False
    assert circuit.to_qasm()==qasm
    if threshold==0:
        assert max(row['created_tags'] for row in observed)>0
        assert max(row['accepted_checks'] for row in observed)>0
    assert all(row['dense_guard_snapshot_allocations']==0 for row in observed)


@pytest.mark.parametrize('kind',['chain','ring','long_range','all_to_all'])
@pytest.mark.parametrize('seed',[421,922])
def test_general_connectivity_full_state(kind,seed):
    circuit=build_general(kind,8,6,seed)
    backend=make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,_=prepare(backend,circuit)
    oracle=NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=3e-11)


@pytest.mark.parametrize('where',[(2,3),(3,2)])
@pytest.mark.parametrize('absorb',['left','right'])
def test_copy_and_alias_contract_with_complex_states(where,absorb,monkeypatch):
    monkeypatch.setattr(AnalyticIdentityRank2MPS,'minimum_merged_bytes',0)
    rng=np.random.default_rng(4221)
    vector=rng.normal(size=64)+1j*rng.normal(size=64);vector/=np.linalg.norm(vector)
    state=AnalyticIdentityRank2MPS(qtn.MatrixProductState.from_dense(vector,[2]*6,cutoff=0))
    for _ in range(4):
        gate=np.eye(4,dtype=complex)[rng.permutation(4)]*np.exp(1j*rng.uniform(-3,3,4))[:,None]
        old=[state[k].data for k in range(6)]
        snapshots=[a.copy() for a in old]
        changed=state.gate_split(gate,where,cutoff=0,max_bond=None,absorb=absorb)
        circuit=Circuit(6);circuit.add(gates.Unitary(gate,*where))
        oracle=NumpyBackend().execute_circuit(circuit,initial_state=vector).state()
        np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1),oracle,atol=3e-11)
        for before,snapshot in zip(old,snapshots):np.testing.assert_array_equal(before,snapshot)
        for site in where:assert not np.shares_memory(old[site],changed[site].data)
        state,vector=changed,oracle


def test_exact_scope_fallback_and_no_global_replacement():
    backend=make_backend()
    backend.configure_tn_simulation(max_bond_dimension=8,svd_cutoff=1e-10)
    circuit=Circuit(5);circuit.add(gates.H(0));circuit.add(gates.TOFFOLI(0,2,4))
    original=exact.ExactKernelCircuitMPS
    state,details=prepare(backend,circuit)
    assert details['engine']=='native'
    assert exact.ExactKernelCircuitMPS is original
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),
        NumpyBackend().execute_circuit(circuit).state(),atol=3e-11)


@pytest.mark.parametrize('absorb',['left','right'])
@pytest.mark.parametrize('reverse',[False,True])
def test_mutation_invalidates_tag_before_following_real_gate(monkeypatch,absorb,reverse):
    monkeypatch.setattr(AnalyticIdentityRank2MPS,'minimum_merged_bytes',0)
    rng=np.random.default_rng(4321)
    vector=rng.normal(size=64)+1j*rng.normal(size=64);vector/=np.linalg.norm(vector)
    state=AnalyticIdentityRank2MPS(qtn.MatrixProductState.from_dense(vector,[2]*6,cutoff=0))
    state._two_site(None,2,3,absorb)
    site=2 if absorb=='right' else 3
    assert state._known_identity(site)==('left' if absorb=='right' else 'right')
    state[site].data.flat[0]+=.37+.19j
    before=np.asarray(state.to_dense()).reshape(-1)
    gate=np.eye(4,dtype=complex)[[0,1,3,2]]
    circuit=Circuit(6);circuit.add(gates.Unitary(gate,*( (3,2) if reverse else (2,3))))
    oracle=NumpyBackend().execute_circuit(circuit,initial_state=before).state()
    state._two_site(gate,2,3,absorb,reverse=reverse)
    assert state.identity_guard_stats['rejected_checks']>0
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=3e-11)


@pytest.mark.parametrize('width,chunk',[(0,1024),(1025,1024),(2,0),(2,2*1024*1024)])
def test_allocation_probe_rejects_budget_expansion(width,chunk):
    from benchmarks.asc_revision_cpu.probe_identity_guard_memory import check
    with pytest.raises(ValueError):check(width,chunk)


def test_allocation_probe_reports_only_guard_memory():
    from benchmarks.asc_revision_cpu.probe_identity_guard_memory import check
    report=check(8,128)
    assert report['quantum_circuit_executed'] is False
    assert report['observed_process_rss'] is False
    assert report['formal_30q_acceptance_complete'] is False
    assert report['publishable_speedup'] is None
    legacy,analytic=report['methods']
    assert legacy['attached_dense_guard_snapshot_bytes']==report['tensor_bytes']
    assert analytic['attached_dense_guard_snapshot_bytes']==0
    assert analytic['live_guard_stats']['max_scan_chunk_bytes']<=128
