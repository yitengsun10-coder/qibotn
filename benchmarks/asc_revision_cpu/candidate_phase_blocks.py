"""Exact research compiler for mixed Hadamard-free Clifford blocks.

The phase-polynomial / linear-map representation is established mathematics,
not a new algorithmic claim: https://arxiv.org/abs/2210.16195 . This local
implementation handles CNOT, CY and CZ and explicitly verifies its complete
operator representation. It does not replace QiboTN/NumPy MPS simulation.
Only a bounded, state-independent routing-score improvement is accepted.
"""
from types import SimpleNamespace
from qibo import gates
from qibotn.backends._linear_cnot import simplify_block
from qibotn.backends._cnot_schedule import _advance, simulate
from benchmarks.asc_revision_cpu.candidate_mixed_schedule import (
    KINDS, optimize_mixed_schedule, schedule,
)


def representation(n, queue):
    """Return A, l, Q for |x> -> i**(l.x + 2*x.Q.x) |A*x>.

    Q is strictly upper triangular over GF(2), l is over Z/4Z. All operations
    are exact Python integer arithmetic, including phases introduced by CY.
    """
    rows = [1 << i for i in range(n)]
    linear = [0] * n
    quadratic = [0] * n
    def phase_on_parity(mask):
        for i in range(n):
            if mask & (1 << i):
                linear[i] = (linear[i] + 1) % 4
                quadratic[i] ^= mask & ~((1 << (i + 1)) - 1)
    def product_phase(a, b):
        for i in range(n):
            ai, bi = bool(a & (1 << i)), bool(b & (1 << i))
            if ai and bi:
                linear[i] = (linear[i] + 2) % 4
            upper = ~((1 << (i + 1)) - 1)
            if ai:
                quadratic[i] ^= b & upper
            if bi:
                quadratic[i] ^= a & upper
    for gate in queue:
        name, qubits = gate.name, gate.qubits
        if name in KINDS and len(qubits) == 2:
            c, t = qubits
            a, b = rows[c], rows[t]
            if name == "cy":
                phase_on_parity(a)
                product_phase(a, b)
            elif name == "cz":
                product_phase(a, b)
            if name != "cz":
                rows[t] ^= rows[c]
        elif name == "s" and len(qubits) == 1:
            phase_on_parity(rows[qubits[0]])
        elif name == "z" and len(qubits) == 1:
            phase_on_parity(rows[qubits[0]])
            phase_on_parity(rows[qubits[0]])
        else:
            raise ValueError(f"Unsupported phase-polynomial gate: {name}")
    return tuple(rows), tuple(linear), tuple(quadratic)


def synthesize(n, block):
    original_representation = representation(n, block)
    _, linear, quadratic = original_representation
    result = []
    for i, value in enumerate(linear):
        if value & 1:
            result.append(gates.S(i))
        if value & 2:
            result.append(gates.Z(i))
    for i, row in enumerate(quadratic):
        for j in range(i + 1, n):
            if row & (1 << j):
                result.append(gates.CZ(i, j))
    pairs = tuple(tuple(g.qubits) for g in block if g.name != "cz")
    result.extend(gates.CNOT(*pair) for pair in simplify_block(n, pairs))
    if representation(n, result) != original_representation:
        raise RuntimeError("Phase compilation changed the complete symbolic operator")
    return tuple(result)


def optimize_phase_blocks(circuit):
    # Compare to the existing exact commutation candidate, not a deliberately
    # slow ordering. Unsupported gates are immutable compilation barriers.
    existing, prior = optimize_mixed_schedule(circuit)
    output, pending, blocks = [], [], []
    order, center = list(range(circuit.nqubits)), 0
    def flush():
        nonlocal order, center
        if not pending:
            return
        original = tuple(pending)
        old_cost = simulate(order, center, tuple(g.qubits for g in original))
        selected, cost = original, old_cost
        metadata = {"original_gates": len(original), "searched": False, "accepted": False}
        if 32 <= len(original) <= 512 and any(g.name in {"cy", "cz"} for g in original):
            compiled = synthesize(circuit.nqubits, original)
            # Order the diagonal commuting region before the CNOT region.
            diagonal = [g for g in compiled if len(g.qubits) == 1]
            cz = [g for g in compiled if g.name == "cz"]
            cnot = [g for g in compiled if g.name in {"cx", "cnot"}]
            if cz:
                cz, cz_cost, _ = schedule(cz, order, center)
            else:
                cz_cost = (0, order, center, 0, 0)
            if cnot:
                cnot, _, _ = schedule(cnot, cz_cost[1], cz_cost[2])
            compiled = tuple(diagonal) + tuple(cz) + tuple(cnot)
            pairs = tuple(g.qubits for g in compiled if len(g.qubits) == 2)
            proposed_cost = simulate(order, center, pairs)
            metadata.update({"searched": True, "proposed_gates": len(compiled),
                             "score_before": old_cost[0], "score_proposed": proposed_cost[0],
                             "operator_representation_verified": True})
            # A modest gate-count expansion may trade for fewer expensive
            # tensor swaps. This bound is metadata-only, never output-based.
            if len(compiled) <= 2 * len(original) and proposed_cost[0] < old_cost[0]:
                if representation(circuit.nqubits, compiled) != representation(circuit.nqubits, original):
                    raise RuntimeError("Scheduling broke the phase-polynomial identity")
                selected, cost = compiled, proposed_cost
                metadata["accepted"] = True
        output.extend(selected)
        order, center = cost[1:3]
        blocks.append(metadata)
        pending.clear()
    for gate in existing.queue:
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
        **prior, "research_phase_blocks": blocks,
        "phase_compiler_uses_amplitudes_or_outputs": False,
        "phase_compiler_original_gates": len(circuit.queue),
        "phase_compiler_final_gates": len(output),
        "phase_compiler_basis": "standard Hadamard-free Clifford phase polynomial; locally implemented",
    }
