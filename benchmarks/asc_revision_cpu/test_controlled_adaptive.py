import numpy as np
import pytest
from qibo.backends import NumpyBackend
from threadpoolctl import threadpool_limits,threadpool_info
from benchmarks.asc_revision_cpu.candidate_controlled_adaptive import make_backend,prepare,thread_policy
from benchmarks.asc_revision_cpu.evaluate import build_circuit,observable,dense_expectation


@pytest.mark.parametrize('width,expected',[(32,1),(256,1),(511,1),(512,4),(1023,4),(1024,8),(2047,8),(2048,16)])
def test_policy_depends_only_on_matrix_width(width,expected):
    assert thread_policy(2*width,width)==expected
    assert thread_policy(width,2*width)==expected


@pytest.mark.parametrize('kind',['chain','ring','long_range','all_to_all'])
@pytest.mark.parametrize('seed',[572,903])
def test_independent_state_expectation_and_restored_threads(kind,seed):
    circuit=build_circuit(kind,8,4,seed)
    backend=make_backend();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    with threadpool_limits(limits=2):
        before=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
        state,details=prepare(backend,circuit)
        vector=NumpyBackend().execute_circuit(circuit).state()
        np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),vector,atol=2e-11)
        labels,sites,coefficients=observable(8,'general')
        actual=backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,8)
        assert actual==pytest.approx(dense_expectation(vector,8,labels,sites,coefficients),abs=2e-11)
        assert [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']==before
        assert details['routing_policy']=='six_gate_lookahead'
        assert details['performance_validated'] is False


@pytest.mark.parametrize('kind',['chain','ring','nonlocal_mixed'])
def test_mixed_gates_independent_block_oracle(kind):
    from benchmarks.asc_revision_cpu.supplement30_workloads import build,BlockSumOracle
    circuit=build(kind,7518,3,8)
    backend=make_backend();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,_=prepare(backend,circuit)
    oracle=BlockSumOracle.from_circuit(circuit,3)
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle.dense_small_only(),atol=2e-11)


def test_thread_limits_restored_even_on_exception(monkeypatch):
    import quimb.tensor as qtn
    from benchmarks.asc_revision_cpu.candidate_controlled_adaptive import AdaptiveControlledMPS
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS
    def fail(*args,**kwargs):raise RuntimeError('test injected failure')
    monkeypatch.setattr(ControlledRank2MPS,'_two_site',fail)
    state=AdaptiveControlledMPS(qtn.MPS_rand_state(4,bond_dim=2,dtype='complex128',seed=9))
    with threadpool_limits(limits=2):
        before=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
        with pytest.raises(RuntimeError,match='injected failure'):
            state._two_site(None,1,2,'right')
        assert [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']==before
