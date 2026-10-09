import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from benchmarks.asc_revision_cpu.candidate_phase_blocks import (
    optimize_phase_blocks, representation, synthesize,
)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n", [3, 4, 5])
def test_full_unitary_not_just_one_input_state(n, seed):
    rng = np.random.default_rng(20261100 + seed)
    original = Circuit(n)
    for _ in range(80):
        pair = tuple(int(q) for q in rng.choice(n, 2, replace=False))
        original.add((gates.CNOT, gates.CY, gates.CZ)[int(rng.integers(3))](*pair))
    compiled = synthesize(n, original.queue)
    actual = Circuit(n)
    actual.add(compiled)
    assert representation(n, original.queue) == representation(n, compiled)
    np.testing.assert_array_equal(actual.unitary(backend=NumpyBackend()),
                                  original.unitary(backend=NumpyBackend()))


@pytest.mark.parametrize("seed", [20261111, 20261112, 20261113])
def test_nonclifford_barriers_and_full_state(seed):
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    original = build("mixed_entanglers", seed, 8, 4)
    qasm = original.to_qasm()
    compiled, metadata = optimize_phase_blocks(original)
    actual = Circuit(8)
    actual.add(compiled.queue)
    backend = NumpyBackend()
    np.testing.assert_allclose(backend.execute_circuit(actual).state(),
                               backend.execute_circuit(original).state(), atol=1e-12, rtol=1e-12)
    assert original.to_qasm() == qasm
    assert metadata["phase_compiler_uses_amplitudes_or_outputs"] is False
    assert [g for g in actual.queue if g.name in {"ry", "rz"}] == [
        g for g in original.queue if g.name in {"ry", "rz"}]


def test_reject_unsupported_rotation_in_symbolic_block():
    with pytest.raises(ValueError, match="Unsupported"):
        representation(2, [gates.RY(0, .3)])
