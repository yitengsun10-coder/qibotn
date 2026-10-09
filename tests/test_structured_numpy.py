"""Integrated size-adaptive exact kernels: values, ownership and dispatch."""
import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends._exact_numpy import ExactKernelCircuitMPS, ExactKernelMPS
from qibotn.backends._structured_numpy import StructuredExactKernelMPS
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import build_circuit
from benchmarks.asc_revision_cpu.probe_linear_holdout import build


@pytest.mark.parametrize("kind", ["chain", "ring", "long_range", "all_to_all", "random_cnot", "mixed_entanglers"])
@pytest.mark.parametrize("seed", [20261031, 20261032])
@pytest.mark.parametrize("threshold", [0, 256*1024])
def test_integrated_full_state_and_dispatch(monkeypatch, kind, seed, threshold):
    monkeypatch.setattr(StructuredExactKernelMPS, "minimum_merged_bytes", threshold)
    circuit = build(kind, seed, 8, 5) if kind in {"random_cnot", "mixed_entanglers"} else build_circuit(kind, 8, 6, seed)
    original = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, details = backend._prepare_optimized_mps(circuit)
    assert isinstance(state, StructuredExactKernelMPS) == (threshold == 0)
    expected = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1e-11, rtol=1e-11)
    values = []
    for method in ("reference", "shared_environments"):
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0, expectation_method=method)
        values.append(backend.exp_value_observable_symbolic(circuit,
            ["xy", "zz", "x"], [(0, 7), (2, 5), (3,)], [1., -.3, .7], 8))
        engine = backend.last_expectation_diagnostics["state_preparation"]["engine"]
        assert engine == ("native_reference" if method == "reference" else "routed_exact")
    np.testing.assert_allclose(*values, atol=1e-11, rtol=1e-11)
    assert original == circuit.to_qasm()


@pytest.mark.parametrize("seed", [21, 49, 110])
@pytest.mark.parametrize("absorb", ["left", "right"])
@pytest.mark.parametrize("where", [(2, 3), (3, 2)])
def test_monomial_on_arbitrary_complex_input(monkeypatch, seed, absorb, where):
    monkeypatch.setattr(StructuredExactKernelMPS, "minimum_merged_bytes", 0)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=64)+1j*rng.normal(size=64)
    vector /= np.linalg.norm(vector)
    matrix = np.eye(4, dtype=complex)[rng.permutation(4)]*np.exp(1j*rng.uniform(-3, 3, 4))[:, None]
    state = StructuredExactKernelMPS(qtn.MatrixProductState.from_dense(vector, [2]*6, cutoff=0))
    original = state.to_dense().copy()
    changed = state.gate_split(matrix, where, cutoff=0, max_bond=None, absorb=absorb)
    circuit = Circuit(6)
    circuit.add(gates.Unitary(matrix, *where))
    expected = NumpyBackend().execute_circuit(circuit, initial_state=vector).state()
    np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1), expected, atol=1e-11)
    np.testing.assert_array_equal(state.to_dense(), original)


def test_mutated_data_invalidates_identity_metadata():
    state = StructuredExactKernelMPS(qtn.MPS_computational_state("000"))
    state._put(0, state._layout(0), np.eye(2, dtype=complex).reshape(1, 2, 2), "left")
    assert state._known_identity(0) == "left"
    state[0].data.flat[0] = .5
    assert state._known_identity(0) is None


def test_replaced_data_and_copy_do_not_reuse_stale_metadata():
    state = StructuredExactKernelMPS(qtn.MPS_computational_state("000"))
    state._put(0, state._layout(0), np.eye(2, dtype=complex).reshape(1, 2, 2), "left")
    copied = state.copy()
    assert copied._identity_gauges is not state._identity_gauges
    state[0].modify(data=state[0].data.copy())
    assert state._known_identity(0) is None


@pytest.mark.parametrize("qubits,structured", [(12, False), (13, False), (14, True), (18, True)])
def test_circuit_class_uses_only_structural_size_bound(qubits, structured):
    circuit = ExactKernelCircuitMPS(qubits, gate_opts={"cutoff": 0, "max_bond": None})
    assert isinstance(circuit._psi, StructuredExactKernelMPS) == structured


def test_small_merged_tensor_retains_original_kernel(monkeypatch):
    state = StructuredExactKernelMPS(qtn.MPS_computational_state("0000"))
    original = ExactKernelMPS._two_site
    calls = []
    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(ExactKernelMPS, "_two_site", checked)
    state._two_site(None, 1, 2, "right")
    assert calls == [True]
    assert not state._identity_gauges
