import inspect
import numpy as np
import pytest
from threadpoolctl import threadpool_limits,threadpool_info
from quimb.tensor import decomp
from qibotn.backends import _exact_numpy as exact
from qibo.backends import NumpyBackend
from benchmarks.asc_revision_cpu.common_qr_thread_policy import common_qr_policy
from benchmarks.asc_revision_cpu.upstream_reference import load_reference
from benchmarks.asc_revision_cpu.evaluate import build_circuit,observable,dense_expectation


def test_exact_values_signature_and_restored_functions():
    native,kernel=decomp._SPLIT_FNS['qr'],exact._reduced_qr
    rng=np.random.default_rng(12)
    matrix=rng.normal(size=(128,64))+1j*rng.normal(size=(128,64))
    with threadpool_limits(limits=2):
        before=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
        with common_qr_policy() as calls:
            assert inspect.signature(decomp._SPLIT_FNS['qr'])==inspect.signature(native)
            q,_,r=decomp.array_split(matrix,method='qr',absorb='right',cutoff=0,max_bond=None)
            np.testing.assert_allclose(q@r,matrix,atol=1e-12)
            q,r=exact._reduced_qr(matrix)
            np.testing.assert_allclose(q@r,matrix,atol=1e-12)
            assert calls[1]==2
        assert [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']==before
    assert decomp._SPLIT_FNS['qr'] is native and exact._reduced_qr is kernel


def test_failure_restores_registered_drivers():
    native,kernel=decomp._SPLIT_FNS['qr'],exact._reduced_qr
    with pytest.raises(RuntimeError):
        with common_qr_policy():raise RuntimeError('test injected')
    assert decomp._SPLIT_FNS['qr'] is native and exact._reduced_qr is kernel


@pytest.mark.parametrize('kind',['chain','ring','long_range','all_to_all'])
def test_official_math_unchanged_against_independent_state(kind):
    reference,_=load_reference()
    circuit=build_circuit(kind,7,4,758)
    backend=reference();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    labels,sites,coefficients=observable(7,'general')
    vector=NumpyBackend().execute_circuit(circuit).state()
    expected=dense_expectation(vector,7,labels,sites,coefficients)
    with threadpool_limits(limits=1),common_qr_policy() as calls:
        actual=backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,7)
        assert sum(calls.values())>0
    assert actual==pytest.approx(expected,abs=2e-11)
