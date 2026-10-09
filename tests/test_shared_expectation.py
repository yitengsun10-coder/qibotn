"""Independent dense-reference tests for the CPU expectation optimization."""
import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
import quimb as qu
import quimb.tensor as qtn

from qibotn.backends.quimb import (
    QuimbNumpyBackend, _shared_mps_expectation, _transfer_left, _transfer_right,
)
from types import SimpleNamespace


def circuit_for(topology, n=6, layers=4, seed=817):
    rng = np.random.default_rng(seed)
    circuit = Circuit(n)
    for layer in range(layers):
        for q in range(n):
            circuit.add(gates.RY(q, theta=rng.uniform(-np.pi, np.pi)))
            circuit.add(gates.RZ(q, theta=rng.uniform(-np.pi, np.pi)))
        if topology == "chain":
            edges = [(q, q + 1) for q in range(n - 1)]
        elif topology == "ring":
            edges = [(q, (q + 1) % n) for q in range(n)]
        elif topology == "long_range":
            permutation = rng.permutation(n)
            edges = list(zip(permutation[::2], permutation[1::2]))
        elif topology == "all_to_all":
            edges = [(q, r) for q in range(n) for r in range(q + 1, n)]
        else:
            raise ValueError(topology)
        for q, r in edges:
            circuit.add(gates.CNOT(int(q), int(r)))
    return circuit


def dense_expectation(state, n, labels, sites, coeffs):
    """Independent dense Pauli action, without constructing a 4**n matrix."""
    answer = 0.0
    for opstr, where, coeff in zip(labels, sites, coeffs):
        transformed = state.reshape((2,) * n)
        for label, site in zip(opstr, where):
            transformed = np.moveaxis(
                np.tensordot(np.asarray(qu.pauli(label.upper())), transformed,
                             axes=(1, site)), 0, site
            )
        answer += coeff * np.vdot(state, transformed.reshape(-1)).real
    return float(answer / np.vdot(state, state).real)


@pytest.mark.parametrize("topology", ["chain", "ring", "long_range", "all_to_all"])
@pytest.mark.parametrize("method", ["reference", "shared_environments"])
def test_against_independent_dense(topology, method):
    circuit = circuit_for(topology)
    labels = ["x", "y", "z", "xy", "zy", "xyz", "i", "zz", "xy"]
    sites = [(0,), (3,), (5,), (0, 5), (4, 1), (5, 2, 0), (2,), (1, 4), (0, 5)]
    coeffs = [0.3, -0.6, 0.8, 0.25, -0.1, 0.2, 0.05, 0.15, 0.1]
    state = NumpyBackend().execute_circuit(circuit).state()
    expected = dense_expectation(state, 6, labels, sites, coeffs)
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(
        max_bond_dimension=None, svd_cutoff=0.0, expectation_method=method
    )
    actual = backend.exp_value_observable_symbolic(circuit, labels, sites, coeffs, 6)
    assert actual == pytest.approx(expected, abs=1e-11)
    assert backend.last_expectation_diagnostics["state_norm_squared"] == pytest.approx(1)


@pytest.mark.parametrize("method", ["reference", "shared_environments"])
def test_rejects_lossy_state(method):
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(
        max_bond_dimension=1, svd_cutoff=0.0, expectation_method=method
    )
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    with pytest.raises(ValueError, match="state norm squared"):
        backend.exp_value_observable_symbolic(circuit, ["zz"], [(0, 1)], [1.0], 2)


def test_shared_environments_does_not_mutate_and_handles_one_site():
    state = qtn.MatrixProductState.from_dense(np.array([1, 1j]) / np.sqrt(2), [2])
    before = state.to_dense().copy()
    value, norm = _shared_mps_expectation(state, ["y", "i"], [(0,), (0,)], [1, 0.3])
    assert value / norm == pytest.approx(1.3)
    assert np.array_equal(state.to_dense(), before)


@pytest.mark.parametrize("labels,sites,coeffs", [
    (["zz"], [(0,)], [1]), (["z"], [(2,)], [1]),
    (["zz"], [(0, 0)], [1]), (["a"], [(0,)], [1]),
    (["z"], [(0,)], [1j]), (["z"], [], [1]),
    (["z"], [(0,)], [np.nan]), (["z"], [(0,)], [np.inf]),
    (["z"], [(0,)], [-np.inf]),
])
def test_invalid_observable_is_rejected(labels, sites, coeffs):
    backend = QuimbNumpyBackend()
    with pytest.raises(ValueError):
        backend.exp_value_observable_symbolic(Circuit(2), labels, sites, coeffs, 2)


@pytest.mark.parametrize("method", ["reference", "shared_environments"])
def test_normalizes_scaled_state_without_claiming_truncation_is_accurate(method, monkeypatch):
    vector = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2) * 0.3
    state = qtn.MatrixProductState.from_dense(vector, [2, 2])
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(expectation_method=method, min_state_norm=0.0)
    monkeypatch.setattr(backend, "_qibo_circuit_to_quimb", lambda *a, **kw: SimpleNamespace(psi=state))
    actual = backend.exp_value_observable_symbolic(Circuit(2), ["zz", "yy"], [(0, 1), (0, 1)], [0.7, 0.2], 2)
    assert actual == pytest.approx(0.5)
    assert backend.last_expectation_diagnostics["state_norm_squared"] == pytest.approx(0.09)


def test_generic_tensor_network_fallback_has_same_normalized_result():
    circuit = circuit_for("all_to_all", n=4, layers=2)
    labels, sites, coeffs = ["xz", "y", "i"], [(3, 0), (2,), (1,)], [0.2, -0.3, 0.1]
    exact = NumpyBackend().execute_circuit(circuit).state()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(ansatz="tn", max_bond_dimension=None, svd_cutoff=0.0)
    actual = backend.exp_value_observable_symbolic(circuit, labels, sites, coeffs, 4)
    assert actual == pytest.approx(dense_expectation(exact, 4, labels, sites, coeffs), abs=1e-11)
    assert backend.last_expectation_diagnostics["method"] == "reference"


@pytest.mark.parametrize("shape", [(3, 7), (7, 3)])
@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("label", ["x", "y", "z"])
def test_hermitian_transfer_matches_general_complex_contraction(shape, direction, label):
    """Test both orientations without relying on canonical circuit gauges."""
    rng = np.random.default_rng(9021)
    channels = tuple(np.ascontiguousarray(rng.normal(size=shape)
                     + 1j * rng.normal(size=shape)) for _ in range(2))
    tensor = channels, tuple(matrix.conj().T for matrix in channels)
    dim = shape[0] if direction == "left" else shape[1]
    random = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    environment = random + random.conj().T
    transfer = _transfer_left if direction == "left" else _transfer_right
    original = environment.copy()
    actual = transfer(environment, tensor, label)
    expected = transfer(environment, tensor, np.asarray(qu.pauli(label.upper())))
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-13)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-11)
    assert np.array_equal(environment, original)


def test_random_pauli_sum_with_unsorted_support_and_cancellation():
    rng = np.random.default_rng(61027)
    n = 7
    vector = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    vector /= np.linalg.norm(vector)
    state = qtn.MatrixProductState.from_dense(vector, [2] * n, cutoff=0)
    labels, sites, coefficients = [], [], []
    for length in range(n + 1):
        for _ in range(5):
            labels.append("".join(rng.choice(list("ixyz"), size=length)))
            sites.append(tuple(rng.choice(n, size=length, replace=False)))
            coefficients.append(float(rng.normal()))
    labels += ["yxy", "yxy"]
    sites += [(6, 0, 3), (6, 0, 3)]
    coefficients += [1.7, -1.7]
    expected = dense_expectation(vector, n, labels, sites, coefficients)
    actual, norm = _shared_mps_expectation(state, labels, sites, coefficients)
    assert actual / norm == pytest.approx(expected, abs=1e-11)
    perm = rng.permutation(len(labels))
    reordered, norm2 = _shared_mps_expectation(
        state, [labels[k] for k in perm], [sites[k] for k in perm],
        [coefficients[k] for k in perm],
    )
    assert reordered / norm2 == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("topology", ["chain", "ring", "long_range", "all_to_all"])
def test_fused_preparation_full_state_against_independent_dense(topology):
    circuit = circuit_for(topology, n=7, layers=4, seed=72019)
    original_qasm = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    fused = backend._qibo_circuit_to_quimb(
        circuit, quimb_circuit_type=backend.circuit_ansatz,
        gate_opts={"cutoff": 0.0, "max_bond": None}, fuse_single_qubit_gates=True,
    ).psi
    expected = NumpyBackend().execute_circuit(circuit).state()
    actual = np.asarray(fused.to_dense()).reshape(-1)
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual * np.exp(-1j * np.angle(phase)), expected,
                               atol=1e-11, rtol=1e-11)
    assert circuit.to_qasm() == original_qasm


def test_fusion_gate_order_and_multi_qubit_boundaries():
    circuit = Circuit(3)
    circuit.add([gates.H(0), gates.RX(0, 0.2), gates.RY(0, -0.6), gates.RZ(0, 0.9),
                 gates.S(0), gates.T(0), gates.Y(1), gates.Z(1), gates.RZ(0, -0.3),
                 gates.CNOT(1, 2), gates.RX(1, 0.7), gates.RY(1, 0.4), gates.CZ(0, 2),
                 gates.U3(2, 0.4, -0.2, 0.7), gates.SWAP(0, 1), gates.H(0),
                 gates.TOFFOLI(0, 1, 2), gates.X(2)])
    backend = QuimbNumpyBackend()
    state = backend._qibo_circuit_to_quimb(
        circuit, quimb_circuit_type=backend.circuit_ansatz,
        gate_opts={"cutoff": 0.0, "max_bond": None}, fuse_single_qubit_gates=True,
    ).psi
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),
        NumpyBackend().execute_circuit(circuit).state(), atol=1e-11, rtol=1e-11)
