import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from benchmarks.asc_revision_cpu.candidate_guarded_scaled import (
    GuardedScaledMPS, make_backend, prepare)
from benchmarks.asc_revision_cpu.evaluate import build_circuit, dense_expectation, observable
from qibotn.backends import quimb as production


@pytest.mark.parametrize('topology', ['chain', 'ring', 'long_range', 'all_to_all'])
@pytest.mark.parametrize('seed', [20261091, 20261092])
@pytest.mark.parametrize('force_structured', [False, True])
def test_full_state_current_scheduler_and_dense(monkeypatch, topology, seed, force_structured):
    if force_structured:
        monkeypatch.setattr(GuardedScaledMPS, 'minimum_merged_bytes', 0)
    circuit = build_circuit(topology, 8, 4, seed)
    qasm = circuit.to_qasm()
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    original = production._prepare_optimized_mps
    state, details = prepare(backend, circuit)
    oracle = np.asarray(NumpyBackend().execute_circuit(circuit).state())
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=2e-11)
    labels, sites, coefficients = observable(8, 'all')
    value = backend.exp_value_observable_symbolic(circuit, labels, sites, coefficients, 8)
    assert value == pytest.approx(dense_expectation(oracle, 8, labels, sites, coefficients), abs=2e-11)
    assert details['routing_policy'] == 'six_gate_lookahead'
    assert details['logical_to_position'] == list(range(8))
    assert production._prepare_optimized_mps is original
    assert circuit.to_qasm() == qasm


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('scale', [.5, -2., 1j])
def test_scaled_shift_preserves_arbitrary_state(side, scale):
    rng = np.random.default_rng(1234)
    vector = rng.normal(size=64) + 1j*rng.normal(size=64)
    state = GuardedScaledMPS(qtn.MatrixProductState.from_dense(vector, [2]*6, cutoff=0))
    state.minimum_merged_bytes = 0
    state._two_site(np.eye(4, dtype=complex), 2, 3,
                    'right' if side == 'left' else 'left')
    site = 2 if side == 'left' else 3
    layout = state._layout(site)
    state._put(site, layout, layout[1]*scale, side, scale)
    before = np.asarray(state.to_dense()).reshape(-1)
    if side == 'left':
        state.right_canonize_site(site)
    else:
        state.left_canonize_site(site)
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), before, atol=1e-11)
    assert state.analytic_gauge_shifts == 1


def test_inplace_external_edit_invalidates_generated_tag():
    state = GuardedScaledMPS(qtn.MPS_computational_state('000'))
    layout = state._layout(0)
    state._put(0, layout, np.eye(2, dtype=complex).reshape(1, 2, 2).transpose(0, 2, 1), 'left')
    assert state._known_identity(0) == 'left'
    state[0].data.flat[0] = .5
    assert state._scaled_identity(0) is None


def test_unsupported_gates_retain_native_fallback(monkeypatch):
    monkeypatch.setattr(GuardedScaledMPS, 'minimum_merged_bytes', 0)
    circuit = Circuit(6)
    circuit.add([gates.H(0), gates.RY(5, .36), gates.CY(5, 0), gates.iSWAP(0, 4),
                 gates.fSim(4, 1, .6, -.7), gates.SWAP(3, 0), gates.TOFFOLI(4, 0, 2)])
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, _ = prepare(backend, circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=2e-11)
