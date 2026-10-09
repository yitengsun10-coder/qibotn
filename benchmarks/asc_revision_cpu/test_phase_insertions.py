import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from benchmarks.asc_revision_cpu.candidate_phase_insertions import inserted_synthesis, optimize_phase_insertions


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("seed", range(8))
def test_full_operator(n, seed):
    rng = np.random.default_rng(20261200 + seed)
    original = Circuit(n)
    for _ in range(100):
        pair = tuple(int(q) for q in rng.choice(n, 2, replace=False))
        original.add((gates.CNOT, gates.CY, gates.CZ)[int(rng.integers(3))](*pair))
    compiled, _ = inserted_synthesis(n, original.queue)
    actual = Circuit(n)
    actual.add(compiled)
    np.testing.assert_array_equal(actual.unitary(backend=NumpyBackend()), original.unitary(backend=NumpyBackend()))


@pytest.mark.parametrize("seed", [20261221, 20261222, 20261223])
def test_nonclifford_full_state(seed):
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    original = build("mixed_entanglers", seed, 8, 4)
    compiled, metadata = optimize_phase_insertions(original)
    actual = Circuit(8)
    actual.add(compiled.queue)
    backend = NumpyBackend()
    np.testing.assert_allclose(backend.execute_circuit(actual).state(), backend.execute_circuit(original).state(),
                               atol=1e-12, rtol=1e-12)
    assert metadata["phase_insertion_uses_amplitudes_or_outputs"] is False
