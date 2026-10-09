"""Research: realize diagonal phases on parities visited by CNOT synthesis.

Uses the standard phase-polynomial representation, not a new group-theoretic
algorithm. Exact integer verification is mandatory before MPS simulation.
State/output data never enters synthesis or candidate acceptance.
"""
from types import SimpleNamespace
from qibo import gates
from qibotn.backends._linear_cnot import simplify_block
from qibotn.backends._cnot_schedule import simulate, _advance
from benchmarks.asc_revision_cpu.candidate_mixed_schedule import KINDS, optimize_mixed_schedule
from benchmarks.asc_revision_cpu.candidate_phase_blocks import representation


def quadratic_vector(n, mask):
    return sum(1 << (i * n + j) for i in range(n) if mask & (1 << i)
               for j in range(i + 1, n) if mask & (1 << j))


def inserted_synthesis(n, block):
    target = representation(n, block)
    _, linear, quadratic = target
    pairs = tuple(tuple(g.qubits) for g in block if g.name != "cz")
    cnot = simplify_block(n, pairs)
    rows = [1 << i for i in range(n)]
    visited, unique = [], set()
    for position, (control, target_qubit) in enumerate(cnot):
        rows[target_qubit] ^= rows[control]
        parity = rows[target_qubit]
        if parity not in unique and parity.bit_count() > 1:
            unique.add(parity)
            visited.append((position, target_qubit, parity))
    basis = {}
    for k, (_, _, parity) in enumerate(visited):
        value, expression = quadratic_vector(n, parity), 1 << k
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot][0]
                expression ^= basis[pivot][1]
            else:
                basis[pivot] = value, expression
                break
    residual = sum(row << (i * n) for i, row in enumerate(quadratic))
    selected = 0
    for pivot in sorted(basis, reverse=True):
        if residual & (1 << pivot):
            residual ^= basis[pivot][0]
            selected ^= basis[pivot][1]
    remaining_linear = list(linear)
    insertions = {}
    for k, (position, qubit, parity) in enumerate(visited):
        if selected & (1 << k):
            insertions.setdefault(position, []).append(gates.S(qubit))
            for i in range(n):
                if parity & (1 << i):
                    remaining_linear[i] = (remaining_linear[i] - 1) % 4
    output = []
    for i, value in enumerate(remaining_linear):
        if value & 1:
            output.append(gates.S(i))
        if value & 2:
            output.append(gates.Z(i))
    for i in range(n):
        for j in range(i + 1, n):
            if residual & (1 << (i * n + j)):
                output.append(gates.CZ(i, j))
    for k, pair in enumerate(cnot):
        output.append(gates.CNOT(*pair))
        output.extend(insertions.get(k, ()))
    if representation(n, output) != target:
        raise RuntimeError("Parity phase insertion changed the full symbolic operator")
    return tuple(output), {"available_quadratic_rank": len(basis),
                           "inserted_phase_gates": selected.bit_count(),
                           "residual_cz_gates": residual.bit_count()}


def optimize_phase_insertions(circuit):
    compiled, prior = optimize_mixed_schedule(circuit)
    output, pending, blocks = [], [], []
    order, center = list(range(circuit.nqubits)), 0
    def flush():
        nonlocal order, center
        if not pending:
            return
        original = tuple(pending)
        cost = simulate(order, center, tuple(g.qubits for g in original))
        chosen = original
        metadata = {"searched": False, "accepted": False, "original_gates": len(original)}
        if 32 <= len(original) <= 512 and any(g.name in {"cy", "cz"} for g in original):
            proposal, details = inserted_synthesis(circuit.nqubits, original)
            proposed_cost = simulate(order, center, tuple(g.qubits for g in proposal if len(g.qubits) == 2))
            metadata.update({"searched": True, "proposed_gates": len(proposal),
                             "score_before": cost[0], "score_proposed": proposed_cost[0], **details})
            if len(proposal) <= 2 * len(original) and proposed_cost[0] < cost[0]:
                chosen, cost = proposal, proposed_cost
                metadata["accepted"] = True
        output.extend(chosen)
        order, center = cost[1:3]
        blocks.append(metadata)
        pending.clear()
    for gate in compiled.queue:
        if gate.name in KINDS and len(gate.qubits) == 2:
            pending.append(gate)
        else:
            flush()
            output.append(gate)
            if len(gate.qubits) == 2 and gate.name in {"iswap", "fsim"}:
                order, center, _, _ = _advance(order, center, gate.qubits)
            elif len(gate.qubits) > 1:
                order, center = list(range(circuit.nqubits)), 0
    flush()
    return SimpleNamespace(nqubits=circuit.nqubits, queue=tuple(output)), {
        **prior, "research_phase_insertions": blocks,
        "phase_insertion_uses_amplitudes_or_outputs": False,
        "phase_insertion_gate_count_change": len(output) - len(circuit.queue)}
