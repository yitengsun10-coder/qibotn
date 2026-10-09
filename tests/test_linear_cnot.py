"""Boolean operator equivalence and arbitrary-state quantum checks."""
import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from qibotn.backends._linear_cnot import linear_rows, simplify_block, optimize_cnot_blocks, static_cost
from qibotn.backends.quimb import QuimbNumpyBackend


@pytest.mark.parametrize("seed", range(20))
def test_random_linear_operators_exactly_equal(seed):
    rng = np.random.default_rng(seed + 815711)
    for n in (2, 3, 5, 8, 16, 30):
        for length in (2, 17, 100):
            pairs = [tuple(int(x) for x in rng.choice(n, size=2, replace=False)) for _ in range(length)]
            compiled = simplify_block(n, pairs)
            assert linear_rows(n, pairs) == linear_rows(n, compiled)
            assert len(compiled) <= len(pairs) and static_cost(compiled) <= static_cost(pairs)


@pytest.mark.parametrize("seed", [3021, 82719, 99927])
def test_compiled_operator_on_arbitrary_complex_input(seed):
    rng = np.random.default_rng(seed)
    n = 6
    c = Circuit(n)
    c.add([gates.CNOT(*[int(x) for x in rng.choice(n, 2, replace=False)]) for _ in range(100)])
    view, _ = optimize_cnot_blocks(c)
    compiled = Circuit(n)
    compiled.add(view.queue)
    vector = rng.normal(size=2**n) + 1j*rng.normal(size=2**n)
    vector /= np.linalg.norm(vector)
    b = NumpyBackend()
    np.testing.assert_allclose(b.execute_circuit(c, initial_state=vector).state(),
                               b.execute_circuit(compiled, initial_state=vector).state(), atol=0, rtol=0)


def test_mixed_gate_boundaries_and_original_circuit_unchanged():
    c = Circuit(6)
    c.add([gates.H(0), gates.RY(5, 0.13)])
    c.add([gates.CNOT(q, r) for q in range(6) for r in range(q+1, 6)])
    c.add([gates.CZ(0, 5), gates.RX(2, 0.31), gates.CY(5, 1)])
    c.add([gates.CNOT(4, 1), gates.CNOT(4, 1), gates.CNOT(2, 0)])
    c.add([gates.TOFFOLI(1, 3, 5), gates.Y(4)])
    qasm = c.to_qasm()
    b = QuimbNumpyBackend()
    b.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, stats = b._prepare_optimized_mps(c)
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),
                               NumpyBackend().execute_circuit(c).state(), atol=1e-11, rtol=1e-11)
    assert c.to_qasm() == qasm and stats["linear_compiler"]["rewritten_blocks"] == 2


@pytest.mark.parametrize("mode,cutoff,bond,enabled", [
    ("native", 0, None, False), ("routed_exact", 1e-12, None, False),
    ("routed_exact", 0, 4, False), ("routed_exact", 0, None, True)])
def test_compiler_only_enabled_for_exact_optimized_execution(mode, cutoff, bond, enabled):
    c = Circuit(4)
    c.add([gates.CNOT(0, 3), gates.CNOT(0, 3)])
    b = QuimbNumpyBackend()
    b.configure_tn_simulation(max_bond_dimension=bond, svd_cutoff=cutoff, mps_execution=mode)
    _, stats = b._prepare_optimized_mps(c)
    assert stats["linear_compiler"]["enabled"] is enabled


def test_explicit_opt_out_preserves_native_gate_count():
    c = Circuit(5)
    c.add([gates.CNOT(q, r) for q in range(5) for r in range(q+1,5)])
    b = QuimbNumpyBackend()
    b.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0, cnot_optimization=False)
    _, stats = b._prepare_optimized_mps(c)
    assert stats["linear_compiler"] == {"enabled": False}


def test_identity_cnot_block_is_removed_without_approximation():
    assert simplify_block(3, [(2,0),(2,0)]) == ()


def test_large_30_qubit_10000_gate_boolean_operator_only():
    # Compiler-stage stress check, not a 30q quantum simulation benchmark.
    rng = np.random.default_rng(910311)
    pairs = [tuple(int(x) for x in rng.choice(30, 2, replace=False)) for _ in range(10000)]
    compiled = simplify_block(30, pairs)
    assert linear_rows(30, pairs) == linear_rows(30, compiled)
    assert len(compiled) <= len(pairs) and static_cost(compiled) <= static_cost(pairs)
