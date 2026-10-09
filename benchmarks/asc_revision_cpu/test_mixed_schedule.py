import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from benchmarks.asc_revision_cpu.candidate_mixed_schedule import (
    _matrix, commutes, optimize_mixed_schedule,
)
from benchmarks.asc_revision_cpu.upstream_reference import load_reference, source_pair


@pytest.mark.parametrize("seed", [20261091, 20261092, 20261093, 20261094])
def test_mixed_complete_operator(seed):
    rng = np.random.default_rng(seed)
    circuit = Circuit(4)
    for _ in range(64):
        pair = tuple(int(q) for q in rng.choice(4, 2, replace=False))
        circuit.add((gates.CNOT, gates.CY, gates.CZ)[int(rng.integers(3))](*pair))
    selected, diagnostics = optimize_mixed_schedule(circuit)
    actual = Circuit(4)
    actual.add(selected.queue)
    backend = NumpyBackend()
    expected_operator = np.asarray(circuit.unitary(backend=backend))
    actual_operator = np.asarray(actual.unitary(backend=backend))
    np.testing.assert_array_equal(actual_operator, expected_operator)
    assert diagnostics["mixed_schedule_gate_count_change"] == 0


@pytest.mark.parametrize("kind,klass", [("x", gates.CNOT), ("y", gates.CY), ("z", gates.CZ)])
def test_local_matrix_against_qibo(kind, klass):
    for pair in ((0, 1), (1, 0), (0, 2), (2, 1)):
        circuit = Circuit(3)
        circuit.add(klass(*pair))
        np.testing.assert_array_equal(_matrix(kind, pair, 3), circuit.unitary(backend=NumpyBackend()))


def test_commutation_is_not_only_shared_control():
    assert commutes(("x", (0, 1)), ("y", (0, 2)))
    assert not commutes(("x", (0, 1)), ("y", (0, 1)))
    assert not commutes(("x", (0, 1)), ("z", (1, 2)))
    assert commutes(("z", (0, 1)), ("z", (1, 2)))


def test_upstream_reference_has_only_typo_change():
    original, repaired = source_pair()
    import difflib
    changes = [line for line in difflib.ndiff(original.splitlines(), repaired.splitlines())
               if line.startswith(("- ", "+ "))]
    assert len(changes) == 2
    assert "optimize=self.contraction_optimizer" in changes[0]
    assert "optimize=self.contractions_optimizer" in changes[1]
    klass, metadata = load_reference()
    assert metadata["extra_norm_contraction"] is False
    backend = klass()
    assert backend.backend == "numpy"
    assert backend.contractions_optimizer == "auto-hq"
