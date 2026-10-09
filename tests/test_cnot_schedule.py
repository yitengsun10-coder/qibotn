"""Integrated bounded commuting-gate scheduling, independent of observables."""
import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends._cnot_schedule import schedule_block, optimize_and_schedule
from qibotn.backends._linear_cnot import linear_rows
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.probe_linear_holdout import build


@pytest.mark.parametrize("n", [4, 8, 12, 18, 30])
@pytest.mark.parametrize("seed", [20261031, 20261032, 20261033])
def test_complete_operator_equivalence(n, seed):
    rng = np.random.default_rng(seed)
    pairs = tuple(tuple(map(int, rng.choice(n, size=2, replace=False))) for _ in range(3*n+30))
    order = list(map(int, rng.permutation(n)))
    before = list(order)
    selected, cost, metadata = schedule_block(n, pairs, order, n//2)
    assert len(selected) == len(pairs)
    assert linear_rows(n, selected) == linear_rows(n, pairs)
    assert cost[0] <= metadata["estimated_score_before"]
    assert order == before


@pytest.mark.parametrize("seed", [20261031, 20261032])
def test_actual_backend_with_searched_blocks(seed):
    circuit = build("random_cnot", seed, 12, 3)
    qasm = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, metadata = backend._prepare_optimized_mps(circuit)
    compiler = metadata["linear_compiler"]
    assert any(b["searched"] for b in compiler["commutation_schedule"])
    assert compiler["schedule_gate_count_change"] == 0
    assert not compiler["schedule_uses_amplitudes_or_outputs"]
    expected = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1e-11)
    assert circuit.to_qasm() == qasm


@pytest.mark.parametrize("seed", [41, 73, 109, 117])
def test_arbitrary_complex_initial_state(seed):
    rng = np.random.default_rng(seed)
    pairs = tuple(tuple(map(int, rng.choice(6, size=2, replace=False))) for _ in range(80))
    selected, _, _ = schedule_block(6, pairs, list(range(6)), 0)
    vector = rng.normal(size=64)+1j*rng.normal(size=64)
    vector /= np.linalg.norm(vector)
    original, reordered = Circuit(6), Circuit(6)
    original.add(gates.CNOT(*pair) for pair in pairs)
    reordered.add(gates.CNOT(*pair) for pair in selected)
    backend = NumpyBackend()
    np.testing.assert_array_equal(backend.execute_circuit(original, initial_state=vector).state(),
        backend.execute_circuit(reordered, initial_state=vector).state())


def test_cnot_optimization_can_be_disabled():
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0, cnot_optimization=False)
    _, metadata = backend._prepare_optimized_mps(build("random_cnot", 921, 8, 2))
    assert metadata["linear_compiler"] == {"enabled": False}


def test_native_barriers_and_short_blocks_are_not_reordered():
    circuit = Circuit(5)
    circuit.add([gates.H(0), gates.CNOT(0, 4), gates.CZ(1, 3), gates.CNOT(4, 2), gates.SWAP(0, 2)])
    view, metadata = optimize_and_schedule(circuit)
    assert tuple(view.queue) == tuple(circuit.queue)
    assert metadata["schedule_gate_count_change"] == 0
