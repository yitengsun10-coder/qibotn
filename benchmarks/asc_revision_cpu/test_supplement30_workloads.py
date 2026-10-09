import numpy as np
import pytest
from qibo.backends import NumpyBackend
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_large_cpu.benchmark import make_observable
from benchmarks.asc_revision_cpu.evaluate import dense_expectation
from benchmarks.asc_revision_cpu.supplement30_workloads import (
    CONFIGURATIONS, BlockSumOracle, build, metadata)


@pytest.mark.parametrize('kind,seed', CONFIGURATIONS)
@pytest.mark.parametrize('group_size', [2, 3])
def test_independent_full_state_and_paulis(kind, seed, group_size):
    circuit = build(kind, seed, group_size, 8)
    oracle = BlockSumOracle.from_circuit(circuit, group_size)
    expected = np.asarray(NumpyBackend().execute_circuit(circuit).state())
    np.testing.assert_allclose(oracle.dense_small_only(), expected, atol=1e-12)
    assert oracle.gates_applied == len(circuit.queue)
    assert len(oracle.factors[0]) == 64
    assert oracle.expectation().real == pytest.approx(1., abs=1e-12)
    n = circuit.nqubits
    for labels, sites in [('xyz', (0, group_size, n-1)), ('yy', (1, n-2)), ('z', (group_size+1,)), ('', ())]:
        value = dense_expectation(expected, n, [labels], [sites], [1.])
        assert oracle.expectation(labels, sites).real == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize('kind,seed', CONFIGURATIONS)
def test_full_mps_overlap_independent_small_scale(kind, seed):
    circuit = build(kind, seed, 3, 8)
    oracle = BlockSumOracle.from_circuit(circuit, 3)
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, _ = backend._prepare_optimized_mps(circuit)
    value, details = oracle.overlap_mps(state)
    dense = np.asarray(state.to_dense()).reshape(-1)
    assert value == pytest.approx(np.vdot(oracle.dense_small_only(), dense), abs=1e-11)
    assert value == pytest.approx(1., abs=1e-11)
    assert details['oracle_branches'] == 64
    assert details['mps_norm_squared'] == pytest.approx(np.vdot(dense, dense).real, abs=1e-11)
    # A wrong site/state must not pass merely because its norm is unchanged.
    state.gate_(np.array([[0., 1.], [1., 0.]]), 7, contract=True)
    changed, _ = oracle.overlap_mps(state)
    assert abs(changed-1) > .01


@pytest.mark.parametrize('kind,seed', CONFIGURATIONS)
def test_predeclared_30q_scale_without_simulating(kind, seed):
    circuit = build(kind, seed)
    description = metadata(circuit, kind, seed)
    assert circuit.nqubits == 30
    assert 9500 <= len(circuit.queue) <= 10100
    assert 400 <= circuit.depth <= 700
    assert description['bridge_gate_count'] == 6
    assert not description['replaces_original30']
    assert description['max_bond'] is None and description['cutoff'] == 0
    with pytest.raises(ValueError, match='Never expand'):
        BlockSumOracle(10).dense_small_only()
