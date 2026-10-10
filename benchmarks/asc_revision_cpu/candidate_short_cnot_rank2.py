"""Research-only bounded short-block scheduling on the same exact CPU MPS.

The production compiler skips CNOT blocks shorter than 32 gates. This
candidate searches blocks of 8..256 gates, based only on gate connectivity
and a simulated canonical-centre travel cost. Noncommuting precedence and
the complete GF(2) operator are checked. No amplitudes, answers, truncation,
gate deletion, circuit-specific pattern or measured timings select an order.
"""
from types import MethodType, SimpleNamespace

from qibo.gates import CNOT
from qibotn.backends import quimb as production
from qibotn.backends._cnot_schedule import _advance, simulate
from qibotn.backends._linear_cnot import linear_rows, optimize_cnot_blocks
from qibotn.backends._mixed_schedule import optimize_mixed_schedule
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import clone, route

MIN_BLOCK = 8
MAX_BLOCK = 256


def schedule_short_block(nqubits, pairs, order, center):
    original = tuple(tuple(pair) for pair in pairs)
    baseline = simulate(order, center, original)
    if not MIN_BLOCK <= len(original) <= MAX_BLOCK:
        return original, baseline, {"searched": False, "gates": len(original)}
    children = [[] for _ in original]
    indegrees = [0] * len(original)
    for j, (control, target) in enumerate(original):
        for i, (earlier_control, earlier_target) in enumerate(original[:j]):
            if earlier_control == target or control == earlier_target:
                children[i].append(j)
                indegrees[j] += 1
    candidates = [(original, tuple(range(len(original))))]
    for weight in (0.5, 2.0, 8.0):
        degrees = list(indegrees)
        ready = {i for i, degree in enumerate(degrees) if not degree}
        active_order, active_center = list(order), center
        indices = []
        while ready:
            positions = {label: site for site, label in enumerate(active_order)}
            def key(index):
                lower, upper = sorted(positions[q] for q in original[index])
                swaps = max(upper - lower - 1, 0)
                travel = max(lower - active_center, active_center - upper, 0)
                return swaps + weight * travel, swaps, index
            chosen = min(ready, key=key)
            ready.remove(chosen)
            indices.append(chosen)
            active_order, active_center, _, _ = _advance(
                active_order, active_center, original[chosen])
            for child in children[chosen]:
                degrees[child] -= 1
                if not degrees[child]:
                    ready.add(child)
        if len(indices) != len(original):
            raise RuntimeError("Incomplete CNOT precedence order")
        candidates.append((tuple(original[i] for i in indices), tuple(indices)))
    scored = [(simulate(order, center, pairs_), index, pairs_, indices)
              for index, (pairs_, indices) in enumerate(candidates)]
    cost, _, selected, indices = min(scored, key=lambda item: (item[0][0], item[1]))
    positions = {old: new for new, old in enumerate(indices)}
    if any(positions[i] >= positions[j] for i, successors in enumerate(children)
           for j in successors):
        raise RuntimeError("Noncommuting precedence changed")
    if linear_rows(nqubits, original) != linear_rows(nqubits, selected):
        raise RuntimeError("Complete CNOT operator changed")
    return selected, cost, {
        "searched": True, "changed": selected != original, "gates": len(original),
        "estimated_score_before": baseline[0], "estimated_score_after": cost[0],
        "estimated_swaps_before": baseline[3], "estimated_swaps_after": cost[3],
        "estimated_center_travel_before": baseline[4],
        "estimated_center_travel_after": cost[4],
        "complete_gf2_operator_verified": True,
        "noncommuting_precedence_verified": True,
    }


def _compile(circuit, base_compiler):
    compiled, details = base_compiler(circuit)
    order, center = list(range(circuit.nqubits)), 0
    output, pending, blocks = [], [], []
    def flush():
        nonlocal order, center
        if not pending:
            return
        pairs = tuple(tuple(gate.qubits) for gate in pending)
        selected, cost, metadata = schedule_short_block(circuit.nqubits, pairs, order, center)
        output.extend(pending if selected == pairs else [CNOT(*pair) for pair in selected])
        order, center = cost[1:3]
        blocks.append(metadata)
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
                order, center = list(range(circuit.nqubits)), 0
    flush()
    if len(output) != len(compiled.queue):
        raise RuntimeError("Scheduling changed compiled gate count")
    return SimpleNamespace(nqubits=circuit.nqubits, queue=tuple(output)), {
        **details, "short_cnot_schedule": blocks, "short_search_bounds": [MIN_BLOCK, MAX_BLOCK],
        "short_schedule_uses_amplitudes_or_outputs": False,
        "short_schedule_gate_count_change": 0,
        "short_schedule_policy": "strictly lower metadata cost; original wins ties",
    }


def compile_linear(circuit):
    return _compile(circuit, optimize_cnot_blocks)


def compile_mixed(circuit):
    return _compile(circuit, optimize_mixed_schedule)


preparation = clone(production._prepare_optimized_mps, {
    "prepare_routed_mps": route, "optimize_and_schedule": compile_linear,
    "optimize_mixed_schedule": compile_mixed,
})


def prepare(backend, circuit):
    state, details = preparation(backend, circuit)
    return state, {**details, "research_candidate": "short_cnot_controlled_rank2",
                   "production_integrated": False}


def make_backend():
    backend = production.QuimbNumpyBackend()
    entry = clone(production.exp_value_observable_symbolic, {"_prepare_optimized_mps": prepare})
    backend.exp_value_observable_symbolic = MethodType(entry, backend)
    return backend
