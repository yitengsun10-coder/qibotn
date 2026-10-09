"""Exact bounded commutation scheduling for mixed CNOT/CY/CZ blocks.

Every noncommuting original pair retains precedence. Commutation is checked
as an exact local matrix identity on at most three qubits. No state, amplitude,
observable, circuit name or measured speed enters the scheduling decision.
Small/pure/unsupported circuits retain the prior CNOT compiler and routing.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np

from qibotn.backends._cnot_schedule import _advance, simulate

KINDS = {"cx": "x", "cnot": "x", "cy": "y", "cz": "z"}


def _matrix(kind, pair, n):
    result = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    control, target = (1 << (n - 1 - q) for q in pair)
    for source in range(1 << n):
        destination, phase = source, 1
        if source & control:
            if kind == "x":
                destination ^= target
            elif kind == "y":
                destination ^= target
                phase = -1j if source & target else 1j
            else:
                phase = -1 if source & target else 1
        result[destination, source] = phase
    return result


@lru_cache(maxsize=256)
def _commutes_pattern(a, b):
    n = max(*a[1], *b[1]) + 1
    first, second = _matrix(*a, n), _matrix(*b, n)
    return bool(np.array_equal(first @ second, second @ first))


def commutes(a, b):
    ka, qa = a
    kb, qb = b
    if not set(qa).intersection(qb):
        return True
    labels = {q: k for k, q in enumerate(dict.fromkeys((*qa, *qb)))}
    return _commutes_pattern((ka, tuple(labels[q] for q in qa)),
                             (kb, tuple(labels[q] for q in qb)))


def schedule(block, order, center):
    descriptors = [(KINDS[g.name], tuple(g.qubits)) for g in block]
    pairs = tuple(item[1] for item in descriptors)
    old_cost = simulate(order, center, pairs)
    if not 32 <= len(block) <= 256:
        return tuple(block), old_cost, {"searched": False}
    children = [[] for _ in block]
    degrees = [0] * len(block)
    constraints = []
    for j, second in enumerate(descriptors):
        for i in range(j):
            if not commutes(descriptors[i], second):
                children[i].append(j)
                degrees[j] += 1
                constraints.append((i, j))
    candidates = [tuple(range(len(block)))]
    for weight in (0., .5, 2.):
        degree = list(degrees)
        ready = {i for i, d in enumerate(degree) if not d}
        active_order, active_center = list(order), center
        selected = []
        while ready:
            positions = {label: site for site, label in enumerate(active_order)}
            def key(i):
                lower, upper = sorted(positions[q] for q in pairs[i])
                distance = max(upper - lower - 1, 0)
                travel = max(lower - active_center, active_center - upper, 0)
                return distance + weight * travel, distance, i
            chosen = min(ready, key=key)
            ready.remove(chosen)
            selected.append(chosen)
            active_order, active_center, _, _ = _advance(
                active_order, active_center, pairs[chosen])
            for child in children[chosen]:
                degree[child] -= 1
                if not degree[child]:
                    ready.add(child)
        if len(selected) != len(block):
            raise RuntimeError("Mixed-gate precedence graph failed")
        candidates.append(tuple(selected))
    cost, _, indices = min(
        ((simulate(order, center, tuple(pairs[i] for i in indices)), k, indices)
         for k, indices in enumerate(candidates)), key=lambda x: (x[0][0], x[1]))
    positions = {original: position for position, original in enumerate(indices)}
    if sorted(indices) != list(range(len(block))) or any(
            positions[i] >= positions[j] for i, j in constraints):
        raise RuntimeError("Mixed schedule changed a noncommuting precedence")
    return tuple(block[i] for i in indices), cost, {
        "searched": True, "changed": indices != candidates[0],
        "noncommuting_precedences_verified": len(constraints),
        "estimated_score_before": old_cost[0], "estimated_score_after": cost[0],
        "estimated_swaps_before": old_cost[3], "estimated_swaps_after": cost[3],
    }


def optimize_mixed_schedule(circuit):
    from qibotn.backends._cnot_schedule import optimize_and_schedule
    eligible = False
    length, mixed = 0, False
    if circuit.nqubits >= 14:
        for gate in (*circuit.queue, None):
            if gate is not None and gate.name in KINDS and len(gate.qubits) == 2:
                length += 1
                mixed = mixed or gate.name in {"cy", "cz"}
            else:
                eligible = eligible or (mixed and 32 <= length <= 256)
                length, mixed = 0, False
    compiled, prior = optimize_and_schedule(circuit)
    if not eligible:
        return compiled, {**prior, "mixed_commutation_schedule": [],
            "mixed_schedule_uses_amplitudes_or_outputs": False,
            "mixed_schedule_gate_count_change": 0,
            "mixed_schedule_policy": "no eligible mixed block at the bounded-search size"}
    output, pending, blocks = [], [], []
    order, center = list(range(circuit.nqubits)), 0
    def flush():
        nonlocal order, center
        if not pending:
            return
        # Pure CNOT blocks have already received their dedicated scheduler.
        if len({g.name for g in pending}) == 1 and pending[0].name in {"cx", "cnot"}:
            selected = tuple(pending)
            cost = simulate(order, center, tuple(g.qubits for g in pending))
        else:
            selected, cost, metadata = schedule(pending, order, center)
            blocks.append(metadata)
        output.extend(selected)
        order, center = cost[1:3]
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
        **prior, "mixed_commutation_schedule": blocks,
        "mixed_schedule_uses_amplitudes_or_outputs": False,
        "mixed_schedule_gate_count_change": len(output) - len(compiled.queue),
    }


