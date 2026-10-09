"""Integrated routing must retain full states and the public output contract."""
import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibotn.backends._exact_numpy import _choose_route
from qibotn.backends.quimb import QuimbNumpyBackend
from benchmarks.asc_revision_cpu.evaluate import build_circuit
from benchmarks.asc_revision_cpu.probe_linear_holdout import build


@pytest.mark.parametrize("kind", ["chain", "ring", "long_range", "all_to_all",
                                  "random_cnot", "mixed_entanglers"])
@pytest.mark.parametrize("seed", [20261022, 20261023])
def test_full_state_and_real_expectation_entry(kind, seed):
    circuit = (build(kind, seed, 8, 5) if kind in {"random_cnot", "mixed_entanglers"}
               else build_circuit(kind, 8, 6, seed))
    qasm = circuit.to_qasm()
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, metadata = backend._prepare_optimized_mps(circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle,
                               atol=1e-11, rtol=1e-11)
    assert metadata["logical_to_position"] == list(range(8))
    assert metadata["routing_policy"] == "six_gate_lookahead"
    labels, sites, coeff = ["xy", "zz", "x"], [(0, 7), (2, 5), (3,)], [1., -.3, .7]
    values = []
    for method in ("reference", "shared_environments"):
        backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0,
                                       expectation_method=method)
        values.append(backend.exp_value_observable_symbolic(circuit, labels, sites, coeff, 8))
        details = backend.last_expectation_diagnostics["state_preparation"]
        assert details["engine"] == ("native_reference" if method == "reference" else "routed_exact")
        if method != "reference":
            assert details["routing_policy"] == "six_gate_lookahead"
    np.testing.assert_allclose(*values, atol=1e-11, rtol=1e-11)
    assert qasm == circuit.to_qasm()


def test_native_fallback_after_both_route_directions():
    circuit = Circuit(6)
    circuit.add([gates.H(0), gates.RY(5, .23), gates.CNOT(5, 0), gates.CY(1, 4),
                 gates.CZ(0, 5), gates.iSWAP(3, 1), gates.fSim(4, 0, .3, -.6),
                 gates.SWAP(0, 3), gates.TOFFOLI(4, 1, 2), gates.U3(5, .2, .4, -.1)])
    backend = QuimbNumpyBackend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, metadata = backend._prepare_optimized_mps(circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=1e-11)
    assert metadata["logical_to_position"] == list(range(6))


def test_route_choice_is_deterministic_and_does_not_modify_order():
    order = list(range(8))
    arguments = (order, 0, 5, [(0, 6), (0, 7)], (4, 4))
    selected = _choose_route(*arguments)
    assert selected in {(5, 1), (0, 4)}
    assert _choose_route(*arguments) == selected
    assert order == list(range(8))
