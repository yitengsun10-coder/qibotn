from types import SimpleNamespace

import numpy as np
import pytest
from quimb.tensor import CircuitMPS
from qibo.backends import NumpyBackend
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import make_backend as plain_backend
from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import make_backend as short_backend
from benchmarks.asc_revision_cpu.common_mps_thread_policy import common_mps_policy
from benchmarks.asc_revision_cpu.common_qr_thread_policy import common_qr_policy
from benchmarks.asc_revision_cpu.evaluate import dense_expectation, observable
from benchmarks.asc_revision_cpu.upstream_reference import load_reference


@pytest.mark.parametrize('bond,qubits,expected', [(1,(0,1),1), (511,(0,1),1),
    (512,(0,1),4), (1024,(0,1),8), (2048,(0,1),16), (4096,(0,),1)])
def test_shape_only_policy_and_restoration(monkeypatch, bond, qubits, expected):
    def fake(self, gate, *args, **kwargs):
        return [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    monkeypatch.setattr(CircuitMPS, '_apply_gate', fake)
    state = SimpleNamespace(_psi=SimpleNamespace(max_bond=lambda:bond))
    gate = SimpleNamespace(qubits=qubits)
    with threadpool_limits(limits=2):
        with common_mps_policy() as calls:
            assert all(t==expected for t in CircuitMPS._apply_gate(state, gate))
            assert calls[expected]==1
        assert all(p['num_threads']==2 for p in threadpool_info() if p['user_api']=='blas')
    assert CircuitMPS._apply_gate is fake


def test_failure_restores_inherited_descriptor():
    before = vars(CircuitMPS).get('_apply_gate')
    owned = '_apply_gate' in vars(CircuitMPS)
    with pytest.raises(RuntimeError):
        with common_mps_policy():
            raise RuntimeError('injected')
    assert ('_apply_gate' in vars(CircuitMPS)) is owned
    assert vars(CircuitMPS).get('_apply_gate') is before


def test_inner_qr_uses_own_width_and_restores_gate_team(monkeypatch):
    from qibotn.backends import _exact_numpy as exact
    matrix = np.arange(32,dtype=float).reshape(8,4)+1j*np.eye(8,4)
    observed = []
    def fake(self, gate, *args, **kwargs):
        observed.append([p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'])
        q,r = exact._reduced_qr(matrix)
        observed.append([p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'])
        return q@r
    monkeypatch.setattr(CircuitMPS,'_apply_gate',fake)
    state = SimpleNamespace(_psi=SimpleNamespace(max_bond=lambda:2048))
    with threadpool_limits(limits=2),common_mps_policy(),common_qr_policy() as calls:
        result = CircuitMPS._apply_gate(state,SimpleNamespace(qubits=(0,1)))
        assert calls[1]==1 and calls[16]==0
    np.testing.assert_allclose(result,matrix,atol=2e-12)
    assert all(all(t==16 for t in teams) for teams in observed)


@pytest.mark.parametrize('method', ['official', 'plain', 'short'])
def test_complete_independent_state_and_scalar_math_unchanged(method):
    circuit = build_circuit(Workload(10,5))
    oracle = NumpyBackend().execute_circuit(circuit).state()
    labels, sites, coefficients = observable(10,'general')
    expected = dense_expectation(oracle,10,labels,sites,coefficients)
    if method=='official':
        backend = load_reference()[0]()
    else:
        backend = {'plain':plain_backend,'short':short_backend}[method]()
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    original = CircuitMPS._apply_gate
    with threadpool_limits(limits=1), common_mps_policy() as calls, common_qr_policy():
        actual = backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,10)
        assert sum(calls.values())>0
        if method!='official':
            from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import prepare as plain_prepare
            from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import prepare as short_prepare
            state,_ = (plain_prepare if method=='plain' else short_prepare)(backend,circuit)
            np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=5e-11)
    assert actual==pytest.approx(expected,abs=5e-11)
    assert CircuitMPS._apply_gate is original


def test_probe_dispatch_uses_new_entry_and_snapshots():
    from benchmarks.asc_revision_cpu import probe_short_mps_policy as probe
    assert probe.suite.__globals__['__file__']==probe.__file__
    assert any(p.name=='common_mps_thread_policy.py' for p in probe.SOURCES)
