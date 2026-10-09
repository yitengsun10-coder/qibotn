"""Independent correctness and gauge contracts for exact isometric kernels."""
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
from qibo import Circuit,gates
from qibo.backends import NumpyBackend
from qibotn.backends._exact_numpy import ExactKernelMPS,_left_isometric_factor
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import build_circuit
from benchmarks.asc_revision_cpu.probe_linear_holdout import build


@pytest.mark.parametrize("shape",[(2,8),(8,8),(64,128),(96,64)])
@pytest.mark.parametrize("dtype",[np.float64,np.complex128])
def test_exact_factor_contract(shape,dtype):
    rng=np.random.default_rng(771)
    data=rng.normal(size=shape)
    if dtype is np.complex128:
        data=data+1j*rng.normal(size=shape)
    source=np.asarray(data,dtype=dtype)
    before=source.copy()
    q,r=_left_isometric_factor(source)
    np.testing.assert_allclose(q@r,source,atol=1e-12)
    np.testing.assert_allclose(q.conj().T@q,np.eye(q.shape[1]),atol=1e-12)
    assert q.dtype==r.dtype==source.dtype
    assert np.array_equal(source,before)


@pytest.mark.parametrize("kind",["chain","ring","long_range","all_to_all","random_cnot","mixed_entanglers"])
@pytest.mark.parametrize("seed",[231,809])
def test_integrated_full_state(kind,seed):
    c=build(kind,seed,8,5) if kind in {"random_cnot","mixed_entanglers"} else build_circuit(kind,8,6,seed)
    qasm=c.to_qasm()
    b=QuimbNumpyBackend()
    b.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,diag=b._prepare_optimized_mps(c)
    expected=NumpyBackend().execute_circuit(c).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),expected,atol=1e-11,rtol=1e-11)
    assert diag["logical_to_position"]==list(range(8))
    assert c.to_qasm()==qasm


@pytest.mark.parametrize("seed",[1,4,109])
@pytest.mark.parametrize("absorb",["left","right"])
@pytest.mark.parametrize("where",[(2,3),(3,2)])
def test_monomial_on_arbitrary_complex_input(seed,absorb,where):
    rng=np.random.default_rng(seed)
    vector=rng.normal(size=64)+1j*rng.normal(size=64)
    vector/=np.linalg.norm(vector)
    matrix=np.eye(4,dtype=complex)[rng.permutation(4)]*np.exp(1j*rng.uniform(-3,3,4))[:,None]
    state=ExactKernelMPS(qtn.MatrixProductState.from_dense(vector,[2]*6,cutoff=0))
    changed=state.gate_split(matrix,where,cutoff=0,max_bond=None,absorb=absorb)
    c=Circuit(6)
    c.add(gates.Unitary(matrix,*where))
    expected=NumpyBackend().execute_circuit(c,initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1),expected,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize("reverse",[False,True])
@pytest.mark.parametrize("absorb",["left","right"])
def test_public_absorb_contract(reverse,absorb):
    state=ExactKernelMPS(qtn.MPS_rand_state(7,4,dtype="complex128",seed=871))
    where=(3,2) if reverse else (2,3)
    changed=state.gate_split(np.asarray(qu.CNOT()),where,cutoff=0,max_bond=None,absorb=absorb)
    if absorb=="right":
        a=changed._layout(2)[1]
        m=a.transpose(0,2,1).reshape(-1,a.shape[1])
        np.testing.assert_allclose(m.conj().T@m,np.eye(m.shape[1]),atol=1e-11)
    else:
        a=changed._layout(3)[1]
        m=a.reshape(a.shape[0],-1)
        np.testing.assert_allclose(m@m.conj().T,np.eye(m.shape[0]),atol=1e-11)


@pytest.mark.parametrize("absorb",["left","right"])
def test_nonunitary_split_does_not_discard_norm(absorb):
    state=ExactKernelMPS(qtn.MPS_rand_state(5,4,dtype="complex128",seed=611))
    before=np.asarray(state.to_dense()).reshape([2]*5)
    operator=np.diag([2.,.7,1.3,.4]).astype(complex)
    expected=(operator@before.reshape(4,-1)).reshape(-1)
    changed=state.gate_split(operator,(0,1),cutoff=0,max_bond=None,absorb=absorb)
    np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1),expected,atol=1e-11)
