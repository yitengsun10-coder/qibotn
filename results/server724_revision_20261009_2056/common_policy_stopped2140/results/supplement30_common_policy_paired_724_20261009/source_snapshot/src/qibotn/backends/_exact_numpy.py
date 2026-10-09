"""Exact NumPy kernels and persistent routing within the Quimb CPU backend.

This operates on Quimb MPS tensors, not a separate state-vector simulator.
Native Quimb routing, canonical-center tracking and MPS objects are retained.
Approximate, cyclic, non-NumPy and unsupported split options use native code.
"""
from functools import partialmethod
from qibo.gates.abstract import ParametrizedGate

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import qr as scipy_qr


def _reduced_qr(matrix):
    """Exact economy QR with a private Fortran workspace for larger matrices.

    The input may be a view of a Quimb tensor, so overwrite is allowed only on
    an owned copy. Tiny matrices retain NumPy to avoid extra dispatch/copy cost.
    No singular values are dropped and no numerical precision is changed.
    """
    if min(matrix.shape) < 64:
        return np.linalg.qr(matrix, mode="reduced")
    workspace = np.array(matrix, order="F", copy=True)
    return scipy_qr(workspace, mode="economic", overwrite_a=True, check_finite=False)


def _left_isometric_factor(matrix):
    """Return Q,R with Q.H@Q=I and Q@R=matrix, without requiring triangular R.

    The exact private MPS kernels use only this isometric contract. For a
    full-row matrix, identity Q and unchanged R avoid unnecessary QR work.
    The bond size remains min(rows, columns), exactly as in economy QR: no
    singular-value cutoff, numerical-rank guess, or dtype conversion is used.
    A private R copy prevents later tensor updates from aliasing the input.
    Tall matrices retain the existing exact QR. Public QR stays triangular.
    """
    rows, columns = matrix.shape
    if rows <= columns:
        return np.eye(rows, dtype=matrix.dtype), matrix.copy()
    return _reduced_qr(matrix)


class ExactKernelMPS(qtn.MatrixProductState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_layouts = {}
        self._kernel_compatible = (not self.cyclic
            and all(isinstance(self[site].data, np.ndarray) and self.phys_dim(site) == 2
                    for site in range(self.L)))

    def _layout(self, site):
        tensor = self[site]
        cached = self._kernel_layouts.get(site)
        if cached is None or cached[0] != tensor.inds:
            left = (self.bond(site - 1, site),) if site else ()
            right = (self.bond(site, site + 1),) if site + 1 < self.L else ()
            ordered = left + right + (self.site_ind(site),)
            axes = tuple(tensor.inds.index(ind) for ind in ordered)
            inverse = tuple(ordered.index(ind) for ind in tensor.inds)
            cached = (tensor.inds, axes, inverse, bool(left), bool(right))
            self._kernel_layouts[site] = cached
        _, axes, _, has_left, has_right = cached
        data = np.transpose(tensor.data, axes)
        shape = (data.shape[0] if has_left else 1,
                 data.shape[int(has_left)] if has_right else 1, 2)
        return tensor, data.reshape(shape), cached

    @staticmethod
    def _store(layout, array):
        tensor, _, (_, _, inverse, has_left, has_right) = layout
        shape = (() if not has_left else (array.shape[0],))
        shape += (() if not has_right else (array.shape[1],)) + (2,)
        tensor.modify(data=np.transpose(array.reshape(shape), inverse))

    def _supports(self, options):
        allowed = {"cutoff", "max_bond", "absorb", "method", "cutoff_mode", "renorm"}
        return (self._kernel_compatible and not (set(options) - allowed)
                and options.get("cutoff", 1e-10) == 0.0
                and options.get("max_bond") is None
                and options.get("absorb") in {"left", "right"}
                and options.get("method", "auto") in {"auto", "qr"}
                and not options.get("renorm"))

    def _two_site(self, G, i, j, absorb, reverse=False):
        lhs, rhs = self._layout(i), self._layout(j)
        a, b = lhs[1], rhs[1]
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Exact kernel requires NumPy arrays")
        left, right = a.shape[0], b.shape[1]
        # theta axes: left, p_i, right, p_j.
        theta = np.tensordot(a, b, axes=(1, 0))
        if G is None:
            # A routing SWAP: exchange physical indices, not logical metadata.
            matrix = theta.transpose(0, 3, 2, 1).reshape(left * 2, right * 2)
        else:
            gate = np.asarray(G).reshape(2, 2, 2, 2)
            if reverse:
                gate = gate.transpose(1, 0, 3, 2)
            physical = theta.transpose(0, 2, 1, 3).reshape(left * right, 4)
            matrix = (physical @ gate.reshape(4, 4).T).reshape(left, right, 2, 2)
            matrix = matrix.transpose(0, 2, 1, 3).reshape(left * 2, right * 2)
        if absorb == "right":
            q, r = _left_isometric_factor(matrix)
            x, y = q, r
        else:
            q, r = _left_isometric_factor(matrix.T)
            x, y = r.T, q.T
        bond = x.shape[1]
        self._store(lhs, x.reshape(left, 2, bond).transpose(0, 2, 1))
        self._store(rhs, y.reshape(bond, right, 2))

    def gate_split(self, G, where, inplace=False, **options):
        if (not isinstance(G, np.ndarray) or self.cyclic or not self._supports(options)
                or len(where) != 2 or abs(where[0] - where[1]) != 1):
            return super().gate_split(G, where, inplace=inplace, **options)
        target = self if inplace else self.copy()
        i, j = sorted(where)
        target._two_site(G, i, j, options["absorb"], reverse=where[0] > where[1])
        return target

    gate_split_ = partialmethod(gate_split, inplace=True)

    def swap_sites_with_compress(self, i, j, info=None, inplace=False, **options):
        if self.cyclic or not self._supports(options) or abs(i - j) != 1:
            return super().swap_sites_with_compress(i, j, info=info, inplace=inplace, **options)
        target = self if inplace else self.copy()
        i, j = sorted((i, j))
        target.canonicalize_((i, j), info=info)
        target._two_site(None, i, j, options["absorb"])
        if info is not None:
            center = i if options["absorb"] == "left" else j
            info["cur_orthog"] = (center, center)
        return target

    swap_sites_with_compress_ = partialmethod(swap_sites_with_compress, inplace=True)

    def left_canonize_site(self, i, bra=None, create_bond=False):
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().left_canonize_site(i, bra=bra, create_bond=create_bond)
        lhs, rhs = self._layout(i), self._layout(i + 1)
        a, b = lhs[1], rhs[1]
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return super().left_canonize_site(i, bra=bra, create_bond=create_bond)
        q, r = _left_isometric_factor(a.transpose(0, 2, 1).reshape(a.shape[0] * 2, a.shape[1]))
        self._store(lhs, q.reshape(a.shape[0], 2, q.shape[1]).transpose(0, 2, 1))
        self._store(rhs, (r @ b.reshape(b.shape[0], -1)).reshape(r.shape[0], b.shape[1], 2))

    def right_canonize_site(self, i, bra=None, create_bond=False):
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().right_canonize_site(i, bra=bra, create_bond=create_bond)
        lhs, rhs = self._layout(i - 1), self._layout(i)
        a, b = lhs[1], rhs[1]
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return super().right_canonize_site(i, bra=bra, create_bond=create_bond)
        q, r = _left_isometric_factor(b.reshape(b.shape[0], b.shape[1] * 2).T)
        self._store(rhs, q.T.reshape(q.shape[1], b.shape[1], 2))
        updated = a.transpose(0, 2, 1).reshape(-1, a.shape[1]) @ r.T
        self._store(lhs, updated.reshape(a.shape[0], 2, r.shape[0]).transpose(0, 2, 1))


class ExactKernelCircuitMPS(qtn.CircuitMPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical exact qubit MPS merged tensors have at most 2**n
        # complex128 entries. Avoid per-gate subclass work entirely when
        # this structural bound cannot amortize guarded snapshot/index maps.
        from qibotn.backends._structured_numpy import StructuredExactKernelMPS
        state_type = (StructuredExactKernelMPS
            if (1 << self._psi.L) * 16 >= StructuredExactKernelMPS.minimum_merged_bytes
            else ExactKernelMPS)
        self._psi = state_type(self._psi)


def _choose_route(order, lower, upper, future_pairs, center):
    """Circuit-only six-gate lookahead; no amplitudes or observables inspected."""
    if center is None or center == "calc":
        lo, hi = lower, upper
    elif isinstance(center, int):
        lo = hi = center
    else:
        lo, hi = min(center), max(center)
    choices = []
    for initial, final, start in ((upper, lower + 1, (upper - 1, upper)),
                                  (lower, upper - 1, (lower, lower + 1))):
        following = list(order)
        label = following.pop(initial)
        following.insert(final, label)
        positions = {label: site for site, label in enumerate(following)}
        future_cost = sum((0.5 ** k) * max(abs(positions[a] - positions[b]) - 1, 0)
                          for k, (a, b) in enumerate(future_pairs))
        center_cost = max(start[0] - hi, lo - start[1], 0)
        choices.append((future_cost + 0.5 * center_cost, center_cost, initial, final))
    return min(choices)[2:]


def prepare_routed_mps(backend, circuit, gate_map, circuit_type=ExactKernelCircuitMPS):
    if backend.backend != "numpy" or backend.ansatz != "mps":
        raise ValueError("Persistent routing requires NumPy MPS")
    if backend.max_bond_dimension is not None or backend.svd_cutoff != 0.0:
        raise ValueError("Persistent routing requires exact untruncated MPS")
    n = circuit.nqubits
    circ = circuit_type(n, gate_opts={"max_bond": None, "cutoff": 0.0})
    position_to_logical = list(range(n))
    logical_to_position = list(range(n))
    pending_site, pending_matrix = None, None
    routing_swaps = 0
    restoration_swaps = 0
    leftward_routes = rightward_routes = 0

    def flush():
        nonlocal pending_site, pending_matrix
        if pending_matrix is not None:
            circ.apply_gate_raw(pending_matrix, (pending_site,))
            pending_site, pending_matrix = None, None

    def move_label(initial, final):
        logical = position_to_logical.pop(initial)
        position_to_logical.insert(final, logical)
        for position, label in enumerate(position_to_logical):
            logical_to_position[label] = position

    def restore():
        nonlocal restoration_swaps
        flush()
        for target in range(n):
            initial = logical_to_position[target]
            if initial != target:
                circ._psi.swap_site_to_(initial, target, info=circ.gate_opts["info"],
                                       max_bond=None, cutoff=0.0)
                restoration_swaps += abs(initial - target)
                move_label(initial, target)

    routable = {"cx", "cnot", "cy", "cz", "iswap", "fsim"}
    # Research subclasses may have their own absorption/center contracts.
    # Preserve their previous native route instead of overriding that policy.
    use_lookahead = circuit_type is ExactKernelCircuitMPS
    queue = tuple(circuit.queue)
    futures, following = [], []
    if use_lookahead:
        for gate in reversed(queue):
            futures.append(tuple(following[:6]))
            if len(gate.qubits) == 2 and gate.name in routable:
                following = [tuple(gate.qubits), *following[:5]]
            elif len(gate.qubits) > 1:
                following = []
        futures.reverse()
    for index, gate in enumerate(queue):
        gate_name = getattr(gate, "name", None)
        name = gate_map.get(gate_name)
        if name == "measure":
            continue  # Match the existing QiboTN expectation entry semantics.
        if name is None:
            raise ValueError(f"Gate {gate_name} not supported in Quimb backend")
        logical_qubits = tuple(gate.qubits)
        physical = tuple(logical_to_position[q] for q in logical_qubits)
        if len(physical) == 1:
            matrix = np.asarray(gate.matrix(backend), dtype=np.complex128)
            if pending_site == physical[0]:
                pending_matrix = matrix @ pending_matrix
            else:
                flush()
                pending_site, pending_matrix = physical[0], matrix
            continue
        flush()
        params = getattr(gate, "parameters", ())
        opts = {"parametrized": True} if (
            isinstance(gate, ParametrizedGate) and getattr(gate, "trainable", True)
        ) else {}
        if len(physical) == 2 and gate_name in routable:
            lower, upper = sorted(physical)
            if use_lookahead:
                if upper > lower + 1:
                    initial, final = _choose_route(position_to_logical, lower, upper,
                        futures[index], circ.gate_opts["info"].get("cur_orthog"))
                    circ._psi.swap_site_to_(initial, final, info=circ.gate_opts["info"],
                                           max_bond=None, cutoff=0.0)
                    routing_swaps += abs(initial - final)
                    rightward_routes += int(initial < final)
                    leftward_routes += int(initial > final)
                    move_label(initial, final)
                    physical = tuple(logical_to_position[q] for q in logical_qubits)
                circ.apply_gate(name, *params, *physical, swap_back=False, **opts)
                # Reversed physical controls use left absorption, so the
                # actual center is where[1], not always the higher site.
                circ.gate_opts["info"]["cur_orthog"] = (physical[1], physical[1])
            else:
                circ.apply_gate(name, *params, *physical, swap_back=False, **opts)
            if not use_lookahead and upper > lower + 1:
                routing_swaps += upper - lower - 1
                move_label(upper, lower + 1)
        else:
            # Native special SWAP / multi-qubit gates have different handlers.
            # Restore first and use the unchanged official gate implementation.
            restore()
            circ.apply_gate(name, *params, *logical_qubits, **opts)
    restore()
    state = circ.psi
    return state, {"routing_swaps": routing_swaps, "restoration_swaps": restoration_swaps,
                   "logical_to_position": logical_to_position,
                   "max_bond_observed": state.max_bond(),
                   "routing_policy": "six_gate_lookahead" if use_lookahead else "native_upper_to_lower",
                   "kernel_policy": "size_adaptive_structured" if
                       type(circ._psi).__name__ == "StructuredExactKernelMPS" else "exact_numpy",
                   "leftward_routes": leftward_routes, "rightward_routes": rightward_routes}
