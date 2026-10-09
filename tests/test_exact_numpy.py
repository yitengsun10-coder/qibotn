"""Independent full-state and compatibility checks for the exact MPS kernels."""
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from qibotn.backends._exact_numpy import ExactKernelMPS
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import build_circuit


@pytest.mark.parametrize("topology", ["chain", "ring", "long_range", "all_to_all"])
@pytest.mark.parametrize("execution", ["native", "exact_kernels", "routed_exact"])
@pytest.mark.parametrize("seed", [61027, 7919])
def test_preparation_matches_independent_full_state(topology, execution, seed):
    circuit = build_circuit(topology, 7, 4, seed)
    qasm = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0.0, mps_execution=execution)
    state, diagnostic = backend._prepare_optimized_mps(circuit)
    actual = np.asarray(state.to_dense()).reshape(-1)
    expected = NumpyBackend().execute_circuit(circuit).state()
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual * np.exp(-1j * np.angle(phase)), expected, atol=1e-11, rtol=1e-11)
    assert np.vdot(actual, actual).real == pytest.approx(1, abs=1e-11)
    assert circuit.to_qasm() == qasm
    if execution == "routed_exact":
        assert diagnostic["logical_to_position"] == list(range(7))


@pytest.mark.parametrize("execution", ["exact_kernels", "routed_exact"])
def test_reversed_controls_parametric_gates_and_native_fallback(execution):
    c = Circuit(3)
    c.add([gates.H(0), gates.Y(2), gates.RX(2, 0.3), gates.CNOT(2, 0),
           gates.CY(0, 1), gates.CZ(2, 1), gates.iSWAP(0, 2),
           gates.fSim(1, 2, 0.3, -0.6), gates.SWAP(2, 0),
           gates.TOFFOLI(2, 0, 1), gates.RZ(1, 0.9), gates.U3(0, 0.2, -0.7, 0.1)])
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0.0, mps_execution=execution)
    state, _ = backend._prepare_optimized_mps(c)
    actual = np.asarray(state.to_dense()).reshape(-1)
    expected = NumpyBackend().execute_circuit(c).state()
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_out_of_place_split_does_not_mutate_state():
    rng = np.random.default_rng(991)
    vector = rng.normal(size=32) + 1j * rng.normal(size=32)
    vector /= np.linalg.norm(vector)
    state = ExactKernelMPS(qtn.MatrixProductState.from_dense(vector, [2] * 5, cutoff=0))
    before = state.to_dense().copy()
    changed = state.gate_split(np.asarray(qu.CNOT()), (4, 3),
                               cutoff=0.0, max_bond=None, absorb="left")
    assert np.array_equal(state.to_dense(), before)
    c = Circuit(5)
    c.add(gates.CNOT(4, 3))
    expected = NumpyBackend().execute_circuit(c, initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1), expected, atol=1e-11, rtol=1e-11)


def test_nonqubit_state_uses_native_split():
    state = ExactKernelMPS(qtn.MPS_rand_state(4, 3, phys_dim=3, seed=37))
    assert not state._kernel_compatible
    before = state.to_dense()
    changed = state.gate_split(np.eye(9), (1, 2), cutoff=0.0, max_bond=None, absorb="right")
    np.testing.assert_allclose(changed.to_dense(), before, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("cutoff,bond", [(1e-12, None), (0.0, 4)])
def test_truncating_runs_do_not_use_new_execution(cutoff, bond):
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=bond, svd_cutoff=cutoff, mps_execution="routed_exact")
    state, diagnostic = backend._prepare_optimized_mps(build_circuit("long_range", 6, 2, 719))
    assert diagnostic["engine"] == "native"
    assert not isinstance(state, ExactKernelMPS)


def test_invalid_execution_is_rejected():
    with pytest.raises(ValueError, match="mps_execution"):
        QuimbNumpyBackend().configure_tn_simulation(mps_execution="unknown")
