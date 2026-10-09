import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit,gates
from qibo.backends import NumpyBackend
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS,make_backend,prepare
from benchmarks.asc_revision_cpu.supplement30_workloads import build,BlockSumOracle
from benchmarks.asc_revision_cpu.evaluate import build_circuit


@pytest.mark.parametrize('kind',['chain','ring','long_range','all_to_all'])
@pytest.mark.parametrize('seed',[417,919])
def test_general_circuit_full_state(kind,seed):
    circuit=build_circuit(kind,8,5,seed)
    backend=make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,details=prepare(backend,circuit)
    oracle=NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=2e-11)
    assert details['routing_policy']=='six_gate_lookahead'


@pytest.mark.parametrize('constructor',[gates.CNOT,gates.CY,gates.CZ])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('absorb',['left','right'])
def test_rank2_arbitrary_neighbour_tensors(constructor,reverse,absorb):
    rng=np.random.default_rng(302)
    # Two internally entangled regions, joined by a small existing bond.
    arrays=[rng.normal(size=(1,2,2))+1j*rng.normal(size=(1,2,2)),
            rng.normal(size=(2,1,2))+1j*rng.normal(size=(2,1,2)),
            rng.normal(size=(1,2,2))+1j*rng.normal(size=(1,2,2)),
            rng.normal(size=(2,1,2))+1j*rng.normal(size=(2,1,2))]
    state=ControlledRank2MPS(arrays,shape='lrp')
    vector=np.asarray(state.to_dense()).reshape(-1)
    where=(2,1) if reverse else (1,2)
    gate=constructor(*where)
    state._two_site(np.asarray(gate.matrix(NumpyBackend())),1,2,absorb,reverse=reverse)
    circuit=Circuit(4);circuit.add(gate)
    expected=NumpyBackend().execute_circuit(circuit,initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),expected,atol=1e-11)
    assert state.controlled_rank2_splits==1
    assert state.bond_size(1,2)==2
    site=1 if absorb=='right' else 2
    a=state._layout(site)[1]
    matrix=a.transpose(0,2,1).reshape(-1,a.shape[1]) if absorb=='right' else a.reshape(a.shape[0],-1).T
    np.testing.assert_allclose(matrix.conj().T@matrix,np.eye(2),atol=1e-12)


@pytest.mark.parametrize('kind',['chain','ring','nonlocal_mixed'])
def test_all_gates_with_independent_block_oracle(kind):
    circuit=build(kind,1741,3,8)
    backend=make_backend();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    state,_=prepare(backend,circuit)
    oracle=BlockSumOracle.from_circuit(circuit,3)
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle.dense_small_only(),atol=2e-11)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('absorb',['left','right'])
def test_general_complex_controlled_unitary_and_input_ownership(reverse,absorb):
    rng=np.random.default_rng(708)
    arrays=[rng.normal(size=shape)+1j*rng.normal(size=shape) for shape in
            [(1,2,2),(2,1,2),(1,2,2),(2,1,2)]]
    state=ControlledRank2MPS(arrays,shape='lrp')
    vector=np.asarray(state.to_dense()).reshape(-1)
    unitary,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    matrix=np.zeros((4,4),dtype=np.complex128)
    matrix[:2,:2]=np.eye(2);matrix[2:,2:]=unitary
    original_matrix=matrix.copy()
    retained=[state[site].data for site in [1,2]]
    snapshots=[a.copy() for a in retained]
    state._two_site(matrix,1,2,absorb,reverse=reverse)
    circuit=Circuit(4)
    circuit.add(gates.Unitary(matrix,*( (2,1) if reverse else (1,2) )))
    expected=NumpyBackend().execute_circuit(circuit,initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),expected,atol=1e-11)
    np.testing.assert_array_equal(matrix,original_matrix)
    for a,snapshot in zip(retained,snapshots):
        np.testing.assert_array_equal(a,snapshot)
    assert state.controlled_rank2_splits==1


@pytest.mark.parametrize('name',['iswap','fsim','swap','almost_controlled'])
@pytest.mark.parametrize('absorb',['left','right'])
def test_noncontrolled_and_numerically_similar_gate_fallback(name,absorb):
    state=ControlledRank2MPS(qtn.MPS_rand_state(4,bond_dim=2,dtype='complex128',seed=90))
    vector=np.asarray(state.to_dense()).reshape(-1)
    gate={'iswap':gates.iSWAP(1,2),'fsim':gates.fSim(1,2,.31,.27),
          'swap':gates.SWAP(1,2),'almost_controlled':gates.CY(1,2)}[name]
    matrix=np.array(gate.matrix(NumpyBackend()),dtype=np.complex128)
    if name=='almost_controlled':
        matrix[0,0]+=1e-14  # No tolerance-based classification/rank guessing.
    state._two_site(matrix,1,2,absorb)
    circuit=Circuit(4);circuit.add(gates.Unitary(matrix,1,2))
    expected=NumpyBackend().execute_circuit(circuit,initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),expected,atol=1e-11)
    assert state.controlled_rank2_splits==0


@pytest.mark.parametrize('kind',['chain','ring','nonlocal_mixed'])
def test_complete_expectation_entry_and_unchanged_input(kind):
    from benchmarks.asc_revision_cpu.evaluate import observable,dense_expectation
    from benchmarks.asc_revision_cpu.supplement30_workloads import input_bytes
    circuit=build(kind,7621,2,6)
    original=input_bytes(circuit)
    backend=make_backend();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    labels,sites,coefficients=observable(6,'general')
    vector=NumpyBackend().execute_circuit(circuit).state()
    expected=dense_expectation(vector,6,labels,sites,coefficients)
    actual=backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,6)
    assert actual==pytest.approx(expected,abs=2e-11)
    assert input_bytes(circuit)==original
