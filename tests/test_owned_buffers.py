"""Integrated exact buffers: full states, public dispatch and mutable inputs."""
import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends._structured_numpy import StructuredExactKernelMPS as MPS
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import build_circuit
from benchmarks.asc_revision_cpu.probe_linear_holdout import build


@pytest.mark.parametrize("kind", ["chain", "ring", "long_range", "all_to_all", "random_cnot", "mixed_entanglers"])
@pytest.mark.parametrize("seed", [20261041, 20261042])
@pytest.mark.parametrize("threshold", [0, 256 * 1024])
def test_full_state_and_public_dispatch(monkeypatch, kind, seed, threshold):
    monkeypatch.setattr(MPS, "minimum_merged_bytes", threshold)
    circuit = (build(kind, seed, 8, 5) if kind in {"random_cnot", "mixed_entanglers"}
               else build_circuit(kind, 8, 6, seed))
    qasm = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, info = backend._prepare_optimized_mps(circuit)
    assert isinstance(state, MPS) == (threshold == 0)
    expected = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1e-11, rtol=1e-11)
    values = []
    for method in ("reference", "shared_environments"):
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0, expectation_method=method)
        values.append(backend.exp_value_observable_symbolic(circuit,
            ["xy", "zz", "y"], [(0, 7), (2, 5), (3,)], [1., -.3, .7], 8))
        engine = backend.last_expectation_diagnostics["state_preparation"]["engine"]
        assert engine == ("native_reference" if method == "reference" else "routed_exact")
    np.testing.assert_allclose(*values, atol=1e-11, rtol=1e-11)
    assert info["routing_policy"] == "six_gate_lookahead"
    assert qasm == circuit.to_qasm()


@pytest.mark.parametrize("seed", [20261043, 20261044, 20261045])
@pytest.mark.parametrize("absorb", ["left", "right"])
@pytest.mark.parametrize("where", [(2, 3), (3, 2)])
def test_arbitrary_complex_input_copy_ownership(monkeypatch, seed, absorb, where):
    monkeypatch.setattr(MPS, "minimum_merged_bytes", 0)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=64) + 1j * rng.normal(size=64)
    vector /= np.linalg.norm(vector)
    state = MPS(qtn.MatrixProductState.from_dense(vector, [2] * 6, cutoff=0))
    for _ in range(5):
        matrix = np.eye(4, dtype=complex)[rng.permutation(4)] * np.exp(1j * rng.uniform(-3, 3, 4))[:, None]
        before = state.to_dense().copy()
        old_data = [state[i].data for i in range(6)]
        snapshots = [array.copy() for array in old_data]
        changed = state.gate_split(matrix, where, cutoff=0, max_bond=None, absorb=absorb)
        circuit = Circuit(6)
        circuit.add(gates.Unitary(matrix, *where))
        expected = NumpyBackend().execute_circuit(circuit, initial_state=vector).state()
        np.testing.assert_allclose(np.asarray(changed.to_dense()).reshape(-1), expected, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(state.to_dense(), before)
        for old, snapshot in zip(old_data, snapshots):
            np.testing.assert_array_equal(old, snapshot)
        # Quimb shallow copies can share unchanged sites, not changed outputs.
        for site in where:
            assert not np.shares_memory(old_data[site], changed[site].data)
        state, vector = changed, expected


@pytest.mark.parametrize("side", ["left", "right"])
def test_identity_guard_rejects_inplace_and_replaced_data(side):
    state = MPS(qtn.MPS_computational_state("000"))
    site = 0 if side == "left" else 2
    shape = (1, 2, 2) if side == "left" else (2, 1, 2)
    state._put(site, state._layout(site), np.eye(2, dtype=complex).reshape(shape), side)
    assert state._known_identity(site) == side
    state[site].data.flat[0] = .5
    assert state._known_identity(site) is None
    state._put(site, state._layout(site), np.eye(2, dtype=complex).reshape(shape), side)
    state[site].modify(data=state[site].data.copy())
    assert state._known_identity(site) is None


def test_disabled_compiler_and_approximate_native_paths():
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0, cnot_optimization=False)
    circuit = build("random_cnot", 20261046, 8, 4)
    state, info = backend._prepare_optimized_mps(circuit)
    assert info["linear_compiler"] == {"enabled": False}
    expected = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1e-11)
    for options in ({"svd_cutoff": 1e-8}, {"max_bond_dimension": 8}, {"mps_execution": "native"}):
        configuration = {"max_bond_dimension": None, "svd_cutoff": 0, "mps_execution": "routed_exact"}
        configuration.update(options)
        backend.configure_tn_simulation(**configuration)
        _, info = backend._prepare_optimized_mps(circuit)
        assert info["engine"] == "native" and info["linear_compiler"] == {"enabled": False}


@pytest.mark.parametrize("gate", [None, np.eye(4, dtype=complex)[[0, 1, 3, 2]]])
@pytest.mark.parametrize("absorb", ["left", "right"])
def test_inplace_reuses_verified_identity_not_old_neighbor(monkeypatch, gate, absorb):
    monkeypatch.setattr(MPS, "minimum_merged_bytes", 0)
    rng = np.random.default_rng(20261048)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    state = MPS(qtn.MatrixProductState.from_dense(vector, [2] * 4, cutoff=0))
    state._two_site(None, 1, 2, absorb)
    identity_site, neighbor_site = (1, 2) if absorb == "right" else (2, 1)
    identity = state[identity_site].data
    snapshot = state._identity_gauges[identity_site][1]
    old_neighbor = state[neighbor_site].data
    old_neighbor_values = old_neighbor.copy()
    before = np.asarray(state.to_dense()).reshape(-1)
    state._two_site(gate, 1, 2, absorb)
    assert state[identity_site].data is identity
    assert state._identity_gauges[identity_site][1] is snapshot
    assert not np.shares_memory(old_neighbor, state[neighbor_site].data)
    np.testing.assert_array_equal(old_neighbor, old_neighbor_values)
    circuit = Circuit(4)
    circuit.add(gates.SWAP(1, 2) if gate is None else gates.Unitary(gate, 1, 2))
    expected = NumpyBackend().execute_circuit(circuit, initial_state=before).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1e-11)
    preserved = state.to_dense().copy()
    old_neighbor.flat[0] += .7
    np.testing.assert_array_equal(state.to_dense(), preserved)
