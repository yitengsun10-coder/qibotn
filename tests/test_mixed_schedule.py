import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends import quimb
from qibotn.backends._mixed_schedule import _matrix, commutes, schedule, optimize_mixed_schedule
from qibotn.backends._cnot_schedule import optimize_and_schedule


@pytest.mark.parametrize("seed", [20261311, 20261312, 20261313, 20261314])
def test_schedule_complete_operator(seed):
    rng = np.random.default_rng(seed)
    original = Circuit(4)
    for _ in range(64):
        pair = tuple(int(q) for q in rng.choice(4, 2, replace=False))
        original.add((gates.CNOT, gates.CY, gates.CZ)[int(rng.integers(3))](*pair))
    selected, _, metadata = schedule(original.queue, list(range(4)), 0)
    actual = Circuit(4)
    actual.add(selected)
    np.testing.assert_array_equal(actual.unitary(backend=NumpyBackend()), original.unitary(backend=NumpyBackend()))
    assert metadata["searched"]


@pytest.mark.parametrize("kind,klass", [("x", gates.CNOT), ("y", gates.CY), ("z", gates.CZ)])
def test_exact_local_commutation_matrix(kind, klass):
    for pair in ((0, 1), (1, 0), (0, 2), (2, 1)):
        circuit = Circuit(3)
        circuit.add(klass(*pair))
        np.testing.assert_array_equal(_matrix(kind, pair, 3), circuit.unitary(backend=NumpyBackend()))
    assert not commutes(("x", (0, 1)), ("y", (0, 1)))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("seed", [20261321, 20261322])
def test_integrated_full_state(enabled, seed):
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    circuit = build("mixed_entanglers", seed, 14, 4)
    qasm = circuit.to_qasm()
    backend = quimb.QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0,
                                   mixed_gate_schedule=enabled)
    state, details = backend._prepare_optimized_mps(circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=1e-11, rtol=1e-11)
    compiler = details["linear_compiler"]
    if enabled:
        assert any(block["searched"] for block in compiler["mixed_commutation_schedule"])
    else:
        assert "mixed_commutation_schedule" not in compiler
    assert circuit.to_qasm() == qasm


def test_small_and_pure_inputs_use_prior_compiler():
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    for circuit in (build("mixed_entanglers", 20261330, 8, 3),
                    build("random_cnot", 20261331, 14, 3)):
        selected, metadata = optimize_mixed_schedule(circuit)
        prior, _ = optimize_and_schedule(circuit)
        assert [(g.name, g.qubits, g.parameters) for g in selected.queue] == [
            (g.name, g.qubits, g.parameters) for g in prior.queue]
        assert metadata["mixed_commutation_schedule"] == []


@pytest.mark.parametrize("cutoff,bond", [(1e-8, None), (0., 16), (1e-8, 16)])
def test_approximate_path_never_uses_scheduler(monkeypatch, cutoff, bond):
    def fail(_):
        raise AssertionError("Mixed scheduler used in a truncated calculation")
    monkeypatch.setattr(quimb, "optimize_mixed_schedule", fail)
    circuit = Circuit(4)
    circuit.add([gates.H(0), gates.CY(0, 3), gates.CZ(1, 2)])
    backend = quimb.QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=bond, svd_cutoff=cutoff,
                                   mixed_gate_schedule=True)
    _, details = backend._prepare_optimized_mps(circuit)
    assert details["linear_compiler"]["enabled"] is False


def test_all_compilation_can_be_disabled(monkeypatch):
    def fail(_):
        raise AssertionError("Compiler was not disabled")
    monkeypatch.setattr(quimb, "optimize_mixed_schedule", fail)
    circuit = Circuit(4)
    circuit.add([gates.H(0), gates.CY(0, 3)])
    backend = quimb.QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0,
                                   cnot_optimization=False, mixed_gate_schedule=True)
    _, details = backend._prepare_optimized_mps(circuit)
    assert details["linear_compiler"]["enabled"] is False
