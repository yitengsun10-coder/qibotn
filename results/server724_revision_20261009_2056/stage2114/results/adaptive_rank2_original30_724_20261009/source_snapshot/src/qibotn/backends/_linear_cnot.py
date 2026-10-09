"""Exact, state- and observable-independent synthesis of CNOT blocks.

No quantum state or observable is consulted. Only exactly equivalent Boolean
linear maps with lower static cost are accepted. This is used solely in the
untruncated NumPy MPS path, where a changed intermediate circuit is safe.
"""
from types import SimpleNamespace

from qibo.gates import CNOT


def linear_rows(n, pairs):
    rows = [1 << k for k in range(n)]
    for control, target in pairs:
        rows[target] ^= rows[control]
    return rows


def elimination_synthesis(n, pairs, reverse=False):
    rows = linear_rows(n, pairs)
    operations, processed = [], set()
    order = range(n - 1, -1, -1) if reverse else range(n)
    for column in order:
        mask = 1 << column
        if not rows[column] & mask:
            pivot = min((k for k in range(n) if k not in processed and rows[k] & mask),
                        key=lambda k: (abs(k - column), k))
            for control, target in ((pivot, column), (column, pivot), (pivot, column)):
                rows[target] ^= rows[control]
                operations.append((control, target))
        for target in range(n):
            if target != column and rows[target] & mask:
                rows[target] ^= rows[column]
                operations.append((column, target))
        processed.add(column)
    if rows != [1 << k for k in range(n)]:
        raise RuntimeError("CNOT block elimination did not produce identity")
    return tuple(reversed(operations))


def static_cost(pairs):
    # Nearest-neighbor swap-and-return proxy; not a prediction or measured time.
    return sum(1 + 2 * max(abs(c - t) - 1, 0) for c, t in pairs)


def _is_forest(n, pairs):
    parents = list(range(n))
    def root(k):
        while parents[k] != k:
            k = parents[k]
        return k
    for control, target in pairs:
        a, b = root(control), root(target)
        if a == b:
            return False
        parents[a] = b
    return True


def _transpose_rows(rows, n):
    return [sum(((rows[i] >> j) & 1) << i for i in range(n)) for j in range(n)]


def _lower_section_elimination(rows, n, width):
    rows = list(rows)
    operations = []
    for start in range(0, n, width):
        end = min(n, start + width)
        mask = (1 << (end - start)) - 1
        patterns = {}
        for target in range(start, n):
            pattern = (rows[target] >> start) & mask
            if not pattern:
                continue
            if pattern not in patterns:
                patterns[pattern] = target
            else:
                control = patterns[pattern]
                rows[target] ^= rows[control]
                operations.append((control, target))
        for column in range(start, end):
            bit = 1 << column
            if not rows[column] & bit:
                pivot = next((i for i in range(column + 1, n) if rows[i] & bit), None)
                if pivot is None:
                    raise RuntimeError("Singular CNOT linear map")
                rows[column] ^= rows[pivot]
                operations.append((pivot, column))
            for target in range(column + 1, n):
                if rows[target] & bit:
                    rows[target] ^= rows[column]
                    operations.append((column, target))
    return rows, operations


def section_synthesis(n, pairs, width=2, reverse=False):
    """Integer-row adaptation of section-pattern elimination by Patel et al.

    Algorithmic reference: https://arxiv.org/abs/quant-ph/0302002, Algorithm 1.
    The published algorithm is not original work here. Candidate widths are
    bounded in ``simplify_block``; no asymptotic optimality is newly claimed.
    """
    if n < 1 or width < 1:
        raise ValueError("CNOT synthesis needs positive size and section width")
    original = tuple(pairs)
    working = tuple((n-1-c, n-1-t) for c, t in original) if reverse else original
    upper, first = _lower_section_elimination(linear_rows(n, working), n, width)
    identity, second = _lower_section_elimination(_transpose_rows(upper, n), n, width)
    if identity != [1 << k for k in range(n)]:
        raise RuntimeError("Section synthesis failed to reach identity")
    # R*A*C=I. Transpose the second elementary operations and invert the first
    # product, respecting the circuit's chronological gate convention.
    result = tuple((t, c) for c, t in second) + tuple(reversed(first))
    if reverse:
        result = tuple((n-1-c, n-1-t) for c, t in result)
    if linear_rows(n, result) != linear_rows(n, original):
        raise RuntimeError("Section CNOT synthesis is not exactly equivalent")
    return result


def _difference_rank(n, pairs):
    # rank(A-I) lower-bounds the number of CNOTs: each elementary CNOT has
    # rank(E-I)=1 and rank(E*B-I) <= 1 + rank(B-I).
    basis = {}
    for i, row in enumerate(linear_rows(n, pairs)):
        value = row ^ (1 << i)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def simplify_block(n, pairs):
    original = tuple(pairs)
    # Bound compilation overhead on already sparse, acyclic blocks. Keeping
    # the original is always safe; no optimality proof is claimed for this rule.
    if len(original) < 2 or _is_forest(n, original):
        return original
    candidates = [original, elimination_synthesis(n, original), elimination_synthesis(n, original, True)]
    acceptable = [c for c in candidates if len(c) <= len(original)]
    chosen = min(acceptable, key=lambda p: (static_cost(p), len(p)))
    if len(original) >= n:
        lower_bound = _difference_rank(n, original)
        # Skip extra synthesis when the incumbent already meets a rigorous
        # gate-count lower bound with all nearest-neighbor gates.
        if not (len(chosen) == static_cost(chosen) == lower_bound):
            for width in range(1, min(4, n) + 1):
                for reverse in (False, True):
                    candidate = section_synthesis(n, original, width, reverse)
                    if len(candidate) <= len(original):
                        acceptable.append(candidate)
            chosen = min(acceptable, key=lambda p: (static_cost(p), len(p)))
    if (static_cost(chosen), len(chosen)) >= (static_cost(original), len(original)):
        return original
    if linear_rows(n, chosen) != linear_rows(n, original):
        raise RuntimeError("Compiled CNOT block is not exactly equivalent")
    return chosen


def optimize_cnot_blocks(circuit):
    output, pending = [], []
    diagnostics = {"synthesis_family": "gaussian_and_section_patterns",
                   "cnot_blocks": 0, "rewritten_blocks": 0, "original_cnot_gates": 0,
                   "compiled_cnot_gates": 0, "static_cost_before": 0, "static_cost_after": 0}
    def flush():
        if not pending:
            return
        pairs = tuple(tuple(g.qubits) for g in pending)
        selected = simplify_block(circuit.nqubits, pairs)
        diagnostics["cnot_blocks"] += 1
        diagnostics["original_cnot_gates"] += len(pairs)
        diagnostics["compiled_cnot_gates"] += len(selected)
        diagnostics["static_cost_before"] += static_cost(pairs)
        diagnostics["static_cost_after"] += static_cost(selected)
        if selected == pairs:
            output.extend(pending)
        else:
            diagnostics["rewritten_blocks"] += 1
            output.extend(CNOT(*pair) for pair in selected)
        pending.clear()
    for gate in circuit.queue:
        if gate.name in {"cx", "cnot"} and len(gate.qubits) == 2:
            pending.append(gate)
        else:
            flush()
            output.append(gate)
    flush()
    diagnostics["original_total_gates"] = len(circuit.queue)
    diagnostics["compiled_total_gates"] = len(output)
    return SimpleNamespace(nqubits=circuit.nqubits, queue=tuple(output)), diagnostics
