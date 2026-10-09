"""Predeclared supplementary exact workloads, never replacing original30.

Three active ten-qubit regions, with three entangling bridge rounds connecting
ALL regions. Internal connectivity varies. Inter-region entanglement is
deliberately bounded and disclosed; this is not generic all-to-all stress.
The independent oracle is an untruncated sum of block product vectors. It
performs every gate, without allocating a 2**30 dense vector or using Quimb.
"""
import hashlib
import json
import numpy as np
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

CONFIGURATIONS = (
    ('chain', 20261101), ('ring', 20261102), ('nonlocal_mixed', 20261103))


def build(kind, seed, group_size=10, layers=111):
    if kind not in dict(CONFIGURATIONS) or group_size < 2 or layers < 4:
        raise ValueError('Unsupported predeclared topology or small dimensions')
    rng = np.random.default_rng(seed)
    n = 3*group_size
    circuit = Circuit(n)
    for site in range(n):
        circuit.add(gates.H(site))
    bridge_layers = {layers//4, layers//2, 3*layers//4}
    for layer in range(layers):
        for site in range(n):
            circuit.add(gates.RX(site, float(rng.uniform(-np.pi, np.pi))))
            circuit.add(gates.RY(site, float(rng.uniform(-np.pi, np.pi))))
        for stage in range(2):
            for group in range(3):
                offset = group*group_size
                if kind == 'nonlocal_mixed':
                    permutation = list(map(int, rng.permutation(group_size)))
                    pairs = list(zip(permutation[::2], permutation[1::2]))
                else:
                    pairs = [(i, i+1) for i in range(stage, group_size-1, 2)]
                    if kind == 'ring' and stage == 1 and group_size % 2 == 0:
                        pairs.append((group_size-1, 0))
                for i, j in pairs:
                    a, b = offset+i, offset+j
                    if kind == 'nonlocal_mixed':
                        choice = int(rng.integers(5))
                        gate = ((gates.CNOT, gates.CY, gates.CZ, gates.iSWAP)[choice](a, b)
                                if choice < 4 else gates.fSim(a, b,
                                    float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))))
                    else:
                        gate = (gates.CZ if kind == 'ring' and (layer+stage)%2 else gates.CNOT)(a, b)
                    circuit.add(gate)
        if layer in bridge_layers:
            for boundary in range(2):
                a, b = (boundary+1)*group_size-1, (boundary+1)*group_size
                if layer % 2:
                    a, b = b, a
                constructor = (gates.CNOT, gates.CY, gates.CZ)[(layer+boundary)%3]
                circuit.add(constructor(a, b))
    return circuit


def input_bytes(circuit):
    # Qibo's OpenQASM exporter does not support fSim. This complete, canonical
    # gate record preserves every qubit, angle and gate without decomposition.
    record = {'qubits': circuit.nqubits, 'gates': [
        {'name': g.name, 'qubits': list(g.qubits),
         'parameters': list(map(float, getattr(g, 'parameters', ())))}
        for g in circuit.queue]}
    return json.dumps(record, sort_keys=True, separators=(',', ':')).encode('utf-8')


def metadata(circuit, kind, seed, group_size=10):
    counts, edges, bridges = {}, set(), 0
    for gate in circuit.queue:
        counts[gate.name] = counts.get(gate.name, 0)+1
        if len(gate.qubits) == 2:
            a, b = gate.qubits
            edges.add(tuple(sorted((a, b))))
            bridges += int(a//group_size != b//group_size)
    return {'kind': kind, 'seed': seed, 'qubits': circuit.nqubits, 'gates': len(circuit.queue),
            'depth': circuit.depth, 'input_sha256': hashlib.sha256(input_bytes(circuit)).hexdigest(),
            'input_format': 'canonical JSON gate record; fSim is not supported by Qibo OpenQASM exporter',
            'gate_counts': counts, 'distinct_two_qubit_edges': len(edges), 'bridge_gate_count': bridges,
            'group_size': group_size, 'structure_limit': 'three regions with six controlled bridge gates; bounded inter-region entanglement, not generic30',
            'precision': 'complex128', 'cutoff': 0, 'max_bond': None,
            'replaces_original30': False}


def apply_block(vectors, matrix, sites, group_size):
    """Apply a local gate to every branch, in Qibo's big-endian site order."""
    sites = tuple(sites)
    rest = tuple(i for i in range(group_size) if i not in sites)
    axes = (0, *(i+1 for i in sites), *(i+1 for i in rest))
    shaped = vectors.reshape((len(vectors),)+(2,)*group_size).transpose(axes)
    updated = np.einsum('ij,bjk->bik', matrix,
                        shaped.reshape(len(vectors), 1 << len(sites), -1), optimize=True)
    return updated.reshape((len(vectors),)+(2,)*group_size).transpose(np.argsort(axes)).reshape(vectors.shape)


class BlockSumOracle:
    def __init__(self, group_size=10):
        self.group_size = group_size
        self.factors = [np.eye(1, 1 << group_size, dtype=np.complex128) for _ in range(3)]
        self.gates_applied = 0

    def apply(self, gate, backend):
        groups = {site//self.group_size for site in gate.qubits}
        if len(groups) == 1:
            group = next(iter(groups))
            local = tuple(site % self.group_size for site in gate.qubits)
            self.factors[group] = apply_block(self.factors[group],
                np.asarray(gate.matrix(backend), dtype=np.complex128), local, self.group_size)
        else:
            if gate.name not in {'cx', 'cy', 'cz'}:
                raise ValueError('Independent bridge expansion only supports CX/CY/CZ')
            control, target = gate.qubits
            c, t = control//self.group_size, target//self.group_size
            p0, p1 = np.diag([1., 0.]), np.diag([0., 1.])
            pauli = {'cx': np.array([[0., 1.], [1., 0.]]),
                     'cy': np.array([[0., -1j], [1j, 0.]]), 'cz': np.diag([1., -1.])}[gate.name]
            before = self.factors
            after = []
            for group, vectors in enumerate(before):
                branch0, branch1 = vectors, vectors
                if group == c:
                    branch0 = apply_block(vectors, p0, (control % self.group_size,), self.group_size)
                    branch1 = apply_block(vectors, p1, (control % self.group_size,), self.group_size)
                if group == t:
                    branch1 = apply_block(vectors, pauli, (target % self.group_size,), self.group_size)
                after.append(np.concatenate((branch0, branch1)))
            self.factors = after
        self.gates_applied += 1

    @classmethod
    def from_circuit(cls, circuit, group_size=10):
        oracle, backend = cls(group_size), NumpyBackend()
        for gate in circuit.queue:
            oracle.apply(gate, backend)
        return oracle

    def expectation(self, labels=(), sites=()):
        gram = np.ones((len(self.factors[0]),)*2, dtype=np.complex128)
        pauli = {'i': np.eye(2), 'x': np.array([[0., 1.], [1., 0.]]),
                 'y': np.array([[0., -1j], [1j, 0.]]), 'z': np.diag([1., -1.])}
        for group, vectors in enumerate(self.factors):
            changed = vectors
            for label, site in zip(labels, sites):
                if site//self.group_size == group:
                    changed = apply_block(changed, pauli[label.lower()],
                                          (site % self.group_size,), self.group_size)
            gram *= vectors.conj() @ changed.T
        return complex(np.sum(gram))

    def dense_small_only(self):
        if 3*self.group_size > 15:
            raise ValueError('Never expand the 30q oracle to a dense state')
        return np.einsum('bi,bj,bk->ijk', *self.factors, optimize=True).reshape(-1)

    def overlap_mps(self, state):
        """Full <independent oracle|MPS>, blocked with NumPy only, no truncation."""
        if state.L != 3*self.group_size or state.cyclic:
            raise ValueError('Unexpected MPS size/boundary')
        branches = len(self.factors[0])
        environment = np.ones((branches, 1), dtype=np.complex128)
        norm_environment = np.ones((1, 1), dtype=np.complex128)
        largest = 0
        for group, vectors in enumerate(self.factors):
            first = group*self.group_size
            left_size = 1 if first == 0 else state.bond_size(first-1, first)
            block = np.eye(left_size, dtype=np.complex128).reshape(left_size, 1, left_size)
            for site in range(first, first+self.group_size):
                tensor = state[site]
                left = [state.bond(site-1, site)] if site else []
                right = [state.bond(site, site+1)] if site+1 < state.L else []
                array = np.asarray(tensor.transpose(*left, *right, state.site_ind(site)).data)
                array = array.reshape((array.shape[0] if left else 1,
                                       array.shape[len(left)] if right else 1, 2))
                block = np.tensordot(block, array, axes=(2, 0)).transpose(0, 1, 3, 2)
                block = block.reshape(left_size, -1, array.shape[1])
                largest = max(largest, block.nbytes)
            transfer = np.einsum('bp,lpr->blr', vectors.conj(), block, optimize=True)
            environment = np.einsum('bl,blr->br', environment, transfer, optimize=True)
            norm_environment = np.einsum('ab,api,bpj->ij', norm_environment,
                                         block.conj(), block, optimize=True)
        return complex(environment.sum()), {'oracle_branches': branches,
            'oracle_factor_bytes': sum(a.nbytes for a in self.factors),
            'largest_blocked_mps_array_bytes': largest, 'gates_applied': self.gates_applied,
            'mps_norm_squared': float(norm_environment[0, 0].real)}
