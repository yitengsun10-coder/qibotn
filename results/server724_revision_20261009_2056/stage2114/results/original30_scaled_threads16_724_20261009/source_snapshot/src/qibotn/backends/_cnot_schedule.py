"""Exact circuit-only CNOT scheduling for the uncapped NumPy MPS path.

Noncommuting pairs keep their chronological precedence. Three bounded greedy
orders are compared only on metadata routing costs, and the complete GF(2)
operator is verified before use. This changes neither gate count nor precision.
The score is a routing heuristic, never a measured speedup or correctness test.
"""
from types import SimpleNamespace
from qibo.gates import CNOT
from qibotn.backends._linear_cnot import optimize_cnot_blocks, linear_rows
from qibotn.backends._exact_numpy import _choose_route


def _advance(order, center, pair, future=()):
    order = list(order)
    positions = {label: site for site, label in enumerate(order)}
    physical = tuple(positions[q] for q in pair)
    lower, upper = sorted(physical)
    moves = travel = 0
    if upper > lower+1:
        initial, final = _choose_route(order, lower, upper, future, (center, center))
        first = initial-1 if initial > final else initial
        travel = max(first-center, center-first-1, 0)
        label = order.pop(initial)
        order.insert(final, label)
        moves = abs(initial-final)
        positions = {label: site for site, label in enumerate(order)}
        physical = tuple(positions[q] for q in pair)
    else:
        travel = max(lower-center, center-upper, 0)
    return order, physical[1], moves, travel


def simulate(order, center, pairs):
    swaps = travel = 0
    for k, pair in enumerate(pairs):
        order, center, moved, shifted = _advance(order, center, pair, pairs[k+1:k+7])
        swaps += moved
        travel += shifted
    return swaps+.5*travel, order, center, swaps, travel


def schedule_block(n, pairs, order, center):
    original = tuple(pairs)
    old_cost = simulate(order, center, original)
    # Bound O(m^2) dependencies and compilation overhead; short/huge blocks
    # remain unchanged rather than obtaining a hidden unbounded search cost.
    if not 32 <= len(original) <= 256:
        return original, old_cost, {"searched": False}
    children = [[] for _ in original]
    degrees = [0]*len(original)
    for j, (c, t) in enumerate(original):
        for i in range(j):
            a, b = original[i]
            if a == t or c == b:
                children[i].append(j)
                degrees[j] += 1
    candidates = [original]
    for center_weight in (0., .5, 2.):
        remaining = list(degrees)
        ready = {i for i, value in enumerate(remaining) if not value}
        active_order, active_center = list(order), center
        selected = []
        while ready:
            positions = {label: site for site, label in enumerate(active_order)}
            def key(i):
                c, t = original[i]
                lower, upper = sorted((positions[c], positions[t]))
                distance = max(upper-lower-1, 0)
                center_distance = max(lower-active_center, active_center-upper, 0)
                return distance+center_weight*center_distance, distance, i
            chosen = min(ready, key=key)
            ready.remove(chosen)
            pair = original[chosen]
            selected.append(pair)
            active_order, active_center, _, _ = _advance(active_order, active_center, pair)
            for child in children[chosen]:
                remaining[child] -= 1
                if not remaining[child]:
                    ready.add(child)
        if len(selected) != len(original):
            raise RuntimeError("CNOT precedence graph failed")
        candidates.append(tuple(selected))
    scored = [(simulate(order, center, candidate), i, candidate) for i, candidate in enumerate(candidates)]
    cost, _, selected = min(scored, key=lambda item: (item[0][0], item[1]))
    if linear_rows(n, original) != linear_rows(n, selected):
        raise RuntimeError("Scheduled CNOT block changed its complete linear operator")
    return selected, cost, {"searched": True, "changed": selected != original,
        "estimated_score_before": old_cost[0], "estimated_score_after": cost[0],
        "estimated_swaps_before": old_cost[3], "estimated_swaps_after": cost[3]}


def optimize_and_schedule(circuit):
    compiled, diagnostics = optimize_cnot_blocks(circuit)
    lengths, consecutive = [], 0
    for gate in compiled.queue:
        if gate.name in {"cx", "cnot"} and len(gate.qubits) == 2:
            consecutive += 1
        else:
            lengths.append(consecutive)
            consecutive = 0
    lengths.append(consecutive)
    if not any(32 <= length <= 256 for length in lengths):
        return compiled, {**diagnostics, "commutation_schedule": [],
            "schedule_uses_amplitudes_or_outputs": False,
            "schedule_gate_count_change": 0,
            "schedule_policy": "no compiled block reaches bounded-search floor"}
    order, center = list(range(circuit.nqubits)), 0
    output, pending, blocks = [], [], []
    def flush():
        nonlocal order, center
        if not pending:
            return
        pairs = tuple(tuple(gate.qubits) for gate in pending)
        selected, cost, metadata = schedule_block(circuit.nqubits, pairs, order, center)
        blocks.append(metadata)
        output.extend(pending if selected == pairs else [CNOT(*pair) for pair in selected])
        order, center = cost[1:3]
        pending.clear()
    for gate in compiled.queue:
        if gate.name in {"cx", "cnot"} and len(gate.qubits) == 2:
            pending.append(gate)
        else:
            flush()
            output.append(gate)
            if len(gate.qubits) == 2 and gate.name in {"cy", "cz", "iswap", "fsim"}:
                order, center, _, _ = _advance(order, center, gate.qubits)
            elif len(gate.qubits) > 1:
                # Native fallback restores all logical labels before applying.
                order = list(range(circuit.nqubits))
                center = 0
    flush()
    return SimpleNamespace(nqubits=circuit.nqubits, queue=tuple(output)), {
        **diagnostics, "commutation_schedule": blocks,
        "schedule_uses_amplitudes_or_outputs": False,
        "schedule_gate_count_change": len(output)-len(compiled.queue)}

