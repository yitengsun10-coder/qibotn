from collections import Counter, OrderedDict
from typing import Optional

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from qibo.config import raise_error
from qibo.gates.abstract import ParametrizedGate
from qibo.models import Circuit

from qibotn.backends.abstract import QibotnBackend
from qibotn.backends._exact_numpy import ExactKernelCircuitMPS, prepare_routed_mps
from qibotn.backends._linear_cnot import optimize_cnot_blocks
from qibotn.backends._cnot_schedule import optimize_and_schedule
from qibotn.backends._mixed_schedule import optimize_mixed_schedule
from qibotn.result import TensorNetworkResult

GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}


def __init__(self, quimb_backend="numpy", contraction_optimizer="auto-hq"):
    super(self.__class__, self).__init__()

    self.name = "qibotn"
    self.platform = "quimb"
    self.backend = quimb_backend

    self.ansatz = None
    self.max_bond_dimension = None
    self.svd_cutoff = None
    self.n_most_frequent_states = None

    self.configure_tn_simulation()
    self.setup_backend_specifics(
        quimb_backend=quimb_backend, contractions_optimizer=contraction_optimizer
    )


def configure_tn_simulation(
    self,
    ansatz: str = "mps",
    max_bond_dimension: Optional[int] = None,
    svd_cutoff: Optional[float] = 1e-10,
    n_most_frequent_states: int = 100,
    expectation_method: str = "shared_environments",
    min_state_norm: float = 0.99,
    mps_execution: str = "routed_exact",
    cnot_optimization: bool = True,
    mixed_gate_schedule: bool = True,
):
    """
    Configure tensor network simulation.

    ``ansatz='mps'`` selects an open-boundary MPS; other values select a
    generic tensor-network circuit. ``max_bond_dimension=None`` imposes no
    bond cap, while ``svd_cutoff`` still controls truncation during preparation.
    The shared evaluator supports NumPy MPS and otherwise falls back to the
    reference contraction. ``min_state_norm`` is a loss guard, not an accuracy
    or fidelity certificate. Both evaluators return normalized expectations.
    Mixed CNOT/CY/CZ commutation scheduling is bounded and exact; it is used
    only in the uncapped NumPy routed path. Disable it for a compiler ablation.
    """
    self.ansatz = ansatz
    self.max_bond_dimension = max_bond_dimension
    self.svd_cutoff = svd_cutoff
    self.n_most_frequent_states = n_most_frequent_states
    if expectation_method not in {"reference", "shared_environments"}:
        raise ValueError("Unknown expectation_method")
    if not 0.0 <= min_state_norm <= 1.0:
        raise ValueError("min_state_norm must be between zero and one")
    self.expectation_method = expectation_method
    self.min_state_norm = min_state_norm
    if mps_execution not in {"native", "exact_kernels", "routed_exact"}:
        raise ValueError("Unknown mps_execution")
    self.mps_execution = mps_execution
    self.cnot_optimization = bool(cnot_optimization)
    self.mixed_gate_schedule = bool(mixed_gate_schedule)


def _mps_arrays(state):
    """Read physical-channel matrices and their adjoints from an open MPS.

    The circuit can have any gate connectivity: this routine describes its
    final MPS representation, not a restriction on the circuit's gate graph.
    """
    if state.cyclic:
        raise ValueError("Shared environments require an open-boundary MPS")
    arrays = []
    for site in range(state.L):
        tensor = state[site]
        left = [state.bond(site - 1, site)] if site else []
        right = [state.bond(site, site + 1)] if site + 1 < state.L else []
        data = np.asarray(tensor.transpose(*left, *right, state.site_ind(site)).data)
        shape = (data.shape[0] if left else 1,
                 data.shape[len(left)] if right else 1, 2)
        shaped = data.reshape(shape)
        channels = tuple(np.ascontiguousarray(shaped[:, :, p]) for p in range(2))
        arrays.append((channels, tuple(matrix.conj().T for matrix in channels)))
    return arrays


def _transfer_left(environment, tensor, operator=None):
    """Advance one bra/ket environment without forming a rank-4 transfer."""
    channels, adjoints = tensor
    if operator is None:
        return (adjoints[0] @ environment @ channels[0]
                + adjoints[1] @ environment @ channels[1])
    if isinstance(operator, str):
        # Real-weighted Pauli strings have Hermitian prefix environments.
        # The two off-diagonal products are adjoints, so X/Y need only one.
        # This is an algebraic identity, not a truncation of tensor bonds.
        if operator not in {"x", "y", "z"}:
            raise ValueError("Unknown Pauli label")
        if operator == "z":
            return (adjoints[0] @ environment @ channels[0]
                    - adjoints[1] @ environment @ channels[1])
        cross = adjoints[0] @ environment @ channels[1]
        return cross + cross.conj().T if operator == "x" else 1j * (cross.conj().T - cross)
    result = np.zeros((channels[0].shape[1], channels[0].shape[1]), dtype=np.complex128)
    for bra_physical in range(2):
        for ket_physical in range(2):
            coefficient = operator[bra_physical, ket_physical]
            if coefficient != 0:
                result += coefficient * (
                    adjoints[bra_physical] @ environment @ channels[ket_physical]
                )
    return result


def _transfer_right(environment, tensor, operator=None):
    channels, adjoints = tensor
    if operator is None:
        return (adjoints[0].T @ environment @ channels[0].T
                + adjoints[1].T @ environment @ channels[1].T)
    if isinstance(operator, str):
        if operator not in {"x", "y", "z"}:
            raise ValueError("Unknown Pauli label")
        if operator == "z":
            return (adjoints[0].T @ environment @ channels[0].T
                    - adjoints[1].T @ environment @ channels[1].T)
        cross = adjoints[0].T @ environment @ channels[1].T
        return cross + cross.conj().T if operator == "x" else 1j * (cross.conj().T - cross)
    result = np.zeros((channels[0].shape[0], channels[0].shape[0]), dtype=np.complex128)
    for bra_physical in range(2):
        for ket_physical in range(2):
            coefficient = operator[bra_physical, ket_physical]
            if coefficient != 0:
                result += coefficient * (
                    adjoints[bra_physical].T @ environment @ channels[ket_physical].T
                )
    return result


def _shared_mps_expectation(state, operators_list, sites_list, coeffs_list):
    """Contract Pauli sums with exact, byte-bounded bidirectional reuse.

    Intervals crossing the representation midpoint share left prefixes and
    right suffixes. Other intervals contract from the closer side. This does
    not restrict the circuit gate graph or introduce any new truncation.
    """
    arrays = _mps_arrays(state)
    left = [np.ones((1, 1), dtype=np.complex128)]
    for tensor in arrays:
        left.append(_transfer_left(left[-1], tensor))
    right = [None] * (len(arrays) + 1)
    right[-1] = np.ones((1, 1), dtype=np.complex128)
    for site in range(len(arrays) - 1, -1, -1):
        right[site] = _transfer_right(right[site + 1], arrays[site])
    norm = float(left[-1][0, 0].real)
    midpoint = len(arrays) // 2
    cache = OrderedDict()
    cache_bytes = 0

    def cached(key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def remember(key, value):
        nonlocal cache_bytes
        if value.nbytes > 32 * 2**20:
            return
        previous = cache.pop(key, None)
        if previous is not None:
            cache_bytes -= previous.nbytes
        while cache and cache_bytes + value.nbytes > 32 * 2**20:
            _, evicted = cache.popitem(last=False)
            cache_bytes -= evicted.nbytes
        cache[key] = value
        cache_bytes += value.nbytes

    def prefix_environment(support, boundary):
        if not support:
            return left[boundary]
        key = ("left", support, boundary)
        value = cached(key)
        if value is not None:
            return value
        site, label = support[-1]
        inserted_key = ("left", support, site + 1)
        value = cached(inserted_key)
        if value is None:
            value = _transfer_left(prefix_environment(support[:-1], site), arrays[site], label)
            remember(inserted_key, value)
        for position in range(site + 1, boundary):
            gap_key = ("left", support, position + 1)
            previous = cached(gap_key)
            if previous is None:
                value = _transfer_left(value, arrays[position])
                remember(gap_key, value)
            else:
                value = previous
        return value

    def suffix_environment(support, boundary):
        if not support:
            return right[boundary]
        key = ("right", support, boundary)
        value = cached(key)
        if value is not None:
            return value
        site, label = support[0]
        inserted_key = ("right", support, site)
        value = cached(inserted_key)
        if value is None:
            value = _transfer_right(suffix_environment(support[1:], site + 1), arrays[site], label)
            remember(inserted_key, value)
        for position in range(site - 1, boundary - 1, -1):
            gap_key = ("right", support, position)
            previous = cached(gap_key)
            if previous is None:
                value = _transfer_right(value, arrays[position])
                remember(gap_key, value)
            else:
                value = previous
        return value

    answer = 0.0
    for labels, sites, coefficient in zip(operators_list, sites_list, coeffs_list):
        support = tuple(sorted((site, label.lower()) for site, label in zip(sites, labels)
                               if label.lower() != "i"))
        if not support:
            value = norm
        else:
            if support[0][0] < midpoint <= support[-1][0]:
                split = next(k for k, (site, _) in enumerate(support) if site >= midpoint)
                boundary = midpoint
            elif support[-1][0] < midpoint or len(support) == 1:
                split = len(support) - 1
                boundary = support[-1][0]
            else:
                split = 1
                boundary = support[0][0] + 1
            lhs = prefix_environment(support[:split], boundary)
            rhs = suffix_environment(support[split:], boundary)
            value = np.sum(lhs * rhs).real
        answer += float(np.real(coefficient)) * float(value)
    return answer, norm


@property
def circuit_ansatz(self):
    if self.ansatz == "mps":
        return qtn.CircuitMPS
    return qtn.Circuit


def setup_backend_specifics(
    self, quimb_backend="numpy", contractions_optimizer="auto-hq"
):
    """Setup backend specifics.
    Args:
        quimb_backend: str
            The backend to use for the quimb tensor network simulation.
        contractions_optimizer: str, optional
            The contractions_optimizer to use for the quimb tensor network simulation.
    """
    # this is not really working because it does not change the inheritance
    if quimb_backend == "jax":
        import jax.numpy as jnp

        self.engine = jnp
    elif quimb_backend == "numpy":
        import numpy as np

        self.engine = np
    elif quimb_backend == "torch":
        import torch

        self.engine = torch
    else:
        raise_error(ValueError, f"Unsupported quimb backend: {quimb_backend}")

    self.backend = quimb_backend
    self.contractions_optimizer = contractions_optimizer


def execute_circuit(
    self,
    circuit: Circuit,
    initial_state=None,
    nshots=None,
    return_array=False,
):
    """
    Execute a quantum circuit using the specified tensor network ansatz and initial state.

    Args:
        circuit : QuantumCircuit
            The quantum circuit to be executed.
        initial_state : array-like, optional
            The initial state of the quantum system. Only supported for Matrix Product States (MPS) ansatz.
        nshots : int, optional
            The number of shots for sampling the circuit. If None, no sampling is performed, and the full statevector is used.
        return_array : bool, optional
            If True, returns the statevector as a dense array. Default is False.

    Returns:
        TensorNetworkResult
            An object containing the results of the circuit execution, including:
            - nqubits: Number of qubits in the circuit.
            - backend: The backend used for execution.
            - measures: The measurement frequencies if nshots is specified, otherwise None.
            - measured_probabilities: A dictionary of computational basis states and their probabilities.
            - prob_type: The type of probability computation used (currently "default").
            - statevector: The final statevector as a dense array if return_array is True, otherwise None.

    Raises:
        ValueError
            If an initial state is provided but the ansatz is not "MPS".

    Notes:
        - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
        - If `initial_state` is provided, it must be compatible with the MPS ansatz.
        - The `nshots` parameter enables sampling from the circuit's output distribution. If not specified, the full statevector is computed.
    """
    if initial_state is not None and self.ansatz == "MPS":
        initial_state = qtn.tensor_1d.MatrixProductState.from_dense(
            initial_state, 2
        )  # 2 is the physical dimension
    elif initial_state is not None:
        raise_error(ValueError, "Initial state not None supported only for MPS ansatz.")

    circ_quimb = self.circuit_ansatz.from_openqasm2_str(
        circuit.to_qasm(), psi0=initial_state
    )

    if nshots:
        frequencies = Counter(circ_quimb.sample(nshots))
        main_frequencies = {
            state: count
            for state, count in frequencies.most_common(self.n_most_frequent_states)
        }
        computational_states = list(main_frequencies.keys())
        amplitudes = {
            state: circ_quimb.amplitude(state) for state in computational_states
        }
        measured_probabilities = {
            state: abs(amplitude) ** 2 for state, amplitude in amplitudes.items()
        }
    else:
        frequencies = None
        measured_probabilities = None

    statevector = (
        circ_quimb.to_dense(backend=self.backend, optimize=self.contractions_optimizer)
        if return_array
        else None
    )
    return TensorNetworkResult(
        nqubits=circuit.nqubits,
        backend=self,
        measures=frequencies,
        measured_probabilities=measured_probabilities,
        prob_type="default",
        statevector=statevector,
    )


def _prepare_optimized_mps(self, circuit):
    """Prepare an exact NumPy MPS, or conservatively retain native execution.

    Persistent routing changes intermediate bond structure and is therefore
    never silently used with truncation or a bond cap. All sites are restored
    to their original logical order before observable evaluation.
    """
    exact = (self.backend == "numpy" and self.ansatz == "mps"
             and self.max_bond_dimension is None and self.svd_cutoff == 0.0)
    compiler = {"enabled": False}
    if exact and self.mps_execution != "native" and self.cnot_optimization:
        compiler_function = optimize_and_schedule if self.mps_execution == "routed_exact" else optimize_cnot_blocks
        if self.mps_execution == "routed_exact" and self.mixed_gate_schedule:
            compiler_function = optimize_mixed_schedule
        circuit, compiler = compiler_function(circuit)
        compiler = {"enabled": True, **compiler}
    if exact and self.mps_execution == "routed_exact":
        state, details = prepare_routed_mps(self, circuit, GATE_MAP)
        return state, {"engine": "routed_exact", "linear_compiler": compiler, **details}
    circuit_type = ExactKernelCircuitMPS if exact and self.mps_execution == "exact_kernels" else self.circuit_ansatz
    state = self._qibo_circuit_to_quimb(
        circuit, quimb_circuit_type=circuit_type,
        gate_opts={"max_bond": self.max_bond_dimension, "cutoff": self.svd_cutoff},
        fuse_single_qubit_gates=self.backend == "numpy" and self.ansatz == "mps",
    ).psi
    return state, {"engine": "exact_kernels" if circuit_type is ExactKernelCircuitMPS else "native",
                   "linear_compiler": compiler}


def exp_value_observable_symbolic(
    self, circuit, operators_list, sites_list, coeffs_list, nqubits
):
    """
    Compute the expectation value of a symbolic Hamiltonian on a quantum circuit using tensor network contraction.
    This method takes a Qibo circuit, converts it to a Quimb tensor network circuit, and evaluates the expectation value
    of a Hamiltonian specified by three lists of strings: operators, sites, and coefficients.
    The expectation value is computed by summing the contributions from each term in the Hamiltonian, where each term's
    expectation is calculated applying gates to the tensor network representation of the circuit and contracting it
    with the respective bra.
    Each operator string must act on all different qubits, i.e., for each term, the corresponding sites tuple must contain unique qubit indices.
    Example: operators_list = ['xyz', 'xyz'], sites_list = [(1,2,3), (1,2,3)], coeffs_list = [1, 2]


    Parameters
    ----------
    circuit : qibo.models.Circuit
        The quantum circuit to evaluate, provided as a Qibo circuit object.
    operators_list : list of str
        List of operator strings representing the symbolic Hamiltonian terms.
    sites_list : list of tuple of int
        Tuples each specifying the qubits (sites) the corresponding operator acts on.
    coeffs_list : list of real/complex
        The coefficients for each Hamiltonian term.
    Returns
    -------
    float
        The normalized real expectation value of the Hamiltonian. A severely
        truncated state is rejected according to ``min_state_norm``; normalization
        alone does not make an inaccurate truncated state physically correct.
    """

    if nqubits != circuit.nqubits:
        raise ValueError("nqubits must match the circuit")
    if not (len(operators_list) == len(sites_list) == len(coeffs_list)):
        raise ValueError("Observable term lists must have equal lengths")
    for labels, sites, coeff in zip(operators_list, sites_list, coeffs_list):
        if len(labels) != len(sites):
            raise ValueError("Each Pauli string must match its sites")
        if any(label.lower() not in "ixyz" for label in labels):
            raise ValueError("Only I/X/Y/Z Pauli labels are supported")
        if any(site < 0 or site >= nqubits for site in sites):
            raise ValueError("Observable site outside the circuit")
        if not np.isfinite(coeff) or np.imag(coeff) != 0:
            raise ValueError("Hermitian Pauli observables require finite real coefficients")
        if len(sites) != len(set(sites)):
            raise_error(
                ValueError,
                f"Invalid Hamiltonian term sites {sites}: repeated qubit indices are not allowed "
                "within a single term (e.g. (0,0,0) is invalid).",
            )

    if self.expectation_method == "shared_environments":
        circuit_tn, preparation = _prepare_optimized_mps(self, circuit)
    else:
        circuit_tn = self._qibo_circuit_to_quimb(
            circuit, quimb_circuit_type=self.circuit_ansatz,
            gate_opts={"max_bond": self.max_bond_dimension, "cutoff": self.svd_cutoff},
        ).psi
        preparation = {"engine": "native_reference"}
    shared = (
        self.expectation_method == "shared_environments"
        and self.ansatz == "mps"
        and self.backend == "numpy"
        and not circuit_tn.cyclic
    )
    if shared:
        expectation_value, norm = _shared_mps_expectation(
            circuit_tn, operators_list, sites_list, coeffs_list
        )
    else:
        bra = circuit_tn.copy().H
        norm = float((bra & circuit_tn).contract(optimize=self.contractions_optimizer).real)
        expectation_value = 0.0
        for opstr, sites, coeff in zip(operators_list, sites_list, coeffs_list):
            ket = circuit_tn.copy()
            for label, site in zip(opstr, sites):
                ket.gate_(qu.pauli(label.upper()), site)
            value = (bra & ket).contract(optimize=self.contractions_optimizer).real
            expectation_value += float(np.real(coeff)) * value

    self.last_expectation_diagnostics = {
        "method": "shared_environments" if shared else "reference",
        "state_norm_squared": norm,
        "min_state_norm": self.min_state_norm,
        "normalized": True,
        "state_preparation": preparation,
    }
    if not np.isfinite(norm) or norm <= 0 or norm < self.min_state_norm:
        raise ValueError(
            f"MPS state norm squared {norm:.8g} is below the accepted minimum "
            f"{self.min_state_norm:.8g}; verify truncation against an independent "
            "reference before using this result. Normalization is not an accuracy fix."
        )
    return float(np.real(expectation_value)) / norm


def _qibo_circuit_to_quimb(
    self, qibo_circ, quimb_circuit_type=qtn.Circuit, fuse_single_qubit_gates=False,
    **circuit_kwargs
):
    """
    Convert a Qibo Circuit to a Quimb Circuit. Measurement gates are ignored. If are given gates not supported by Quimb, an error is raised.

    Parameters
    ----------
    qibo_circ : qibo.models.circuit.Circuit
        The circuit to convert.
    quimb_circuit_type : type
        The Quimb circuit class to use (Circuit, CircuitMPS, etc).
    circuit_kwargs : dict
        Extra arguments to pass to the Quimb circuit constructor.

    Returns
    -------
    circ : quimb.tensor.circuit.Circuit
        The converted circuit.
    """
    nqubits = qibo_circ.nqubits
    circ = quimb_circuit_type(nqubits, **circuit_kwargs)
    if fuse_single_qubit_gates and self.backend != "numpy":
        raise ValueError("Single-qubit fusion is NumPy CPU only")
    pending_site, pending_matrix = None, None

    def flush_pending():
        nonlocal pending_site, pending_matrix
        if pending_matrix is not None:
            circ.apply_gate_raw(pending_matrix, (pending_site,))
            pending_site, pending_matrix = None, None

    for gate in qibo_circ.queue:
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure":
            continue
        if quimb_gate_name is None:
            raise_error(ValueError, f"Gate {gate_name} not supported in Quimb backend.")

        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())
        if fuse_single_qubit_gates and len(qubits) == 1:
            # Preserve logical gate order: U_last @ ... @ U_first.
            # No gate reordering across sites or entangling gates is used.
            matrix = np.asarray(gate.matrix(self), dtype=np.complex128)
            if pending_site == qubits[0]:
                pending_matrix = matrix @ pending_matrix
            else:
                flush_pending()
                pending_site, pending_matrix = qubits[0], matrix
            continue
        if fuse_single_qubit_gates:
            flush_pending()

        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if is_parametrized:
            circ.apply_gate(
                quimb_gate_name, *params, *qubits, parametrized=is_parametrized
            )
        else:
            circ.apply_gate(
                quimb_gate_name,
                *params,
                *qubits,
            )
    if fuse_single_qubit_gates:
        flush_pending()
    return circ


CLASSES_ROOTS = {"numpy": "Numpy", "torch": "PyTorch", "jax": "Jax"}

METHODS = {
    "__init__": __init__,
    "configure_tn_simulation": configure_tn_simulation,
    "_prepare_optimized_mps": _prepare_optimized_mps,
    "setup_backend_specifics": setup_backend_specifics,
    "execute_circuit": execute_circuit,
    "exp_value_observable_symbolic": exp_value_observable_symbolic,
    "_qibo_circuit_to_quimb": _qibo_circuit_to_quimb,
    "circuit_ansatz": circuit_ansatz,
}


def _generate_backend(quimb_backend: str = "numpy"):
    bases = (QibotnBackend,)

    if quimb_backend == "numpy":
        from qibo.backends import NumpyBackend

        bases += (NumpyBackend,)
    elif quimb_backend == "torch":
        from qiboml.backends import PyTorchBackend

        bases += (PyTorchBackend,)
    elif quimb_backend == "jax":
        from qiboml.backends import JaxBackend

        bases += (JaxBackend,)
    else:
        raise_error(ValueError, f"Unsupported quimb backend: {quimb_backend}")

    return type(f"Quimb{CLASSES_ROOTS[quimb_backend]}Backend", bases, METHODS)


BACKENDS = {}
for k, v in CLASSES_ROOTS.items():
    backend_name = f"Quimb{v}Backend"
    try:
        backend = _generate_backend(k)
        BACKENDS[k] = backend
        globals()[backend_name] = backend
    except ImportError:
        continue


def __getattr__(name):
    try:
        return BACKENDS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
