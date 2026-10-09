"""Exact section synthesis and private QR implementation correctness."""
import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends._linear_cnot import linear_rows, static_cost, section_synthesis, simplify_block
from qibotn.backends._exact_numpy import _reduced_qr


@pytest.mark.parametrize("n", [2,3,5,8,16,30])
@pytest.mark.parametrize("width", [1,2,3,4])
def test_section_operator_equivalence(n,width):
    rng=np.random.default_rng(401+n)
    for length in (2,17,100,1000):
        pairs=tuple(tuple(int(q) for q in rng.choice(n,2,replace=False)) for _ in range(length))
        for reverse in (False,True):
            result=section_synthesis(n,pairs,width,reverse)
            assert linear_rows(n,result)==linear_rows(n,pairs)
        best=simplify_block(n,pairs)
        assert len(best)<=len(pairs) and static_cost(best)<=static_cost(pairs)


@pytest.mark.parametrize("seed", range(4))
def test_section_on_arbitrary_complete_quantum_state(seed):
    rng=np.random.default_rng(seed)
    v=rng.normal(size=64)+1j*rng.normal(size=64)
    v/=np.linalg.norm(v)
    pairs=tuple(tuple(int(q) for q in rng.choice(6,2,replace=False)) for _ in range(70))
    original,compiled=Circuit(6),Circuit(6)
    original.add([gates.CNOT(*p) for p in pairs])
    compiled.add([gates.CNOT(*p) for p in section_synthesis(6,pairs,width=3)])
    backend=NumpyBackend()
    np.testing.assert_array_equal(backend.execute_circuit(original,initial_state=v).state(),
                                  backend.execute_circuit(compiled,initial_state=v).state())


@pytest.mark.parametrize("shape", [(5,3),(70,64),(64,70),(128,80)])
@pytest.mark.parametrize("complex_input", [False,True])
def test_adaptive_qr_reconstructs_without_input_mutation(shape,complex_input):
    rng=np.random.default_rng(317)
    a=rng.normal(size=shape)
    if complex_input:
        a=a+1j*rng.normal(size=shape)
    before=a.copy()
    q,r=_reduced_qr(a)
    np.testing.assert_array_equal(a,before)
    np.testing.assert_allclose(q@r,a,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(q.conj().T@q,np.eye(min(shape)),atol=1e-11,rtol=1e-11)
    assert q.dtype==a.dtype and r.dtype==a.dtype


def test_section_invalid_width_rejected():
    with pytest.raises(ValueError):
        section_synthesis(3,[(0,2)],width=0)
