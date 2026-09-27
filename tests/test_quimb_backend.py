import copy
import os

import config
import numpy as np
import pytest
import qibo
from qibo import Circuit, gates
from qibo.models import QFT
from qibotn.backends.quimb import QuimbNumpyBackend


def create_init_state(nqubits):
    init_state = np.random.random(2**nqubits) + 1j * np.random.random(2**nqubits)
    init_state = init_state / np.sqrt((np.abs(init_state) ** 2).sum())
    return init_state


def qibo_qft(nqubits, init_state, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = circ_qibo(init_state).state(numpy=True)
    return circ_qibo, state_vec


@pytest.mark.parametrize(
    "nqubits, tolerance, is_mps",
    [(1, 1e-6, True), (2, 1e-6, False), (5, 1e-3, True), (10, 1e-3, False)],
)
def test_eval(nqubits: int, tolerance: float, is_mps: bool):
    """Evaluate circuit with Quimb backend.

    Args:
        nqubits (int): Total number of qubits in the system.
        tolerance (float): Maximum limit allowed for difference in results
        is_mps (bool): True if state is MPS and False for tensor network structure
    """
    # hack quimb to use the correct number of processes
    # TODO: remove completely, or at least delegate to the backend
    # implementation
    os.environ["QUIMB_NUM_PROCS"] = str(os.cpu_count())
    import qibotn.eval_qu

    init_state = create_init_state(nqubits=nqubits)
    init_state_tn = copy.deepcopy(init_state)

    # Test qibo
    qibo.set_backend(backend=config.qibo.backend, platform=config.qibo.platform)

    qibo_circ, result_sv = qibo_qft(nqubits, init_state, swaps=True)

    # Convert to qasm for other backends
    qasm_circ = qibo_circ.to_qasm()

    # Test quimb
    if is_mps:
        gate_opt = {}
        gate_opt["method"] = "svd"
        gate_opt["cutoff"] = 1e-6
        gate_opt["cutoff_mode"] = "abs"
    else:
        gate_opt = None
    result_tn = qibotn.eval_qu.dense_vector_tn_qu(
        qasm_circ, init_state_tn, gate_opt, backend=config.quimb.backend
    ).flatten()

    assert np.allclose(
        result_sv, result_tn, atol=tolerance
    ), "Resulting dense vectors do not match"


def test_quimb_expectation_uses_configured_optimizer():
    """Regression test for the Quimb CPU expectation contraction optimizer."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    backend = QuimbNumpyBackend(contraction_optimizer="auto-hq")
    backend.configure_tn_simulation(
        ansatz="mps", max_bond_dimension=16, svd_cutoff=1e-12
    )
    value = backend.exp_value_observable_symbolic(
        circuit,
        operators_list=["zz"],
        sites_list=[(0, 1)],
        coeffs_list=[1.0],
        nqubits=2,
    )

    assert np.isclose(value, 1.0, atol=1e-10)
