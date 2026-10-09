"""Reject partial rows and verify the fresh pinned-upstream comparison."""
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("upstream_typo_only", "frozen", "candidate", "phase_candidate")


def validate_result(data, summary, methods=METHODS):
    if data.get("status") != "pass" or summary.get("status") != "pass" or summary.get("exit_code") != 0:
        raise ValueError("A complete successful worker and controller are required")
    if summary.get("stop_reason") is not None:
        raise ValueError("A stopped benchmark cannot publish a speedup")
    if data.get("precision") != "complex128" or data.get("cutoff") != 0 or data.get("max_bond") is not None:
        raise ValueError("Expected the predeclared untruncated complex128 comparison")
    reference = data["reference"]
    if reference.get("extra_norm_contraction") is not False or reference.get("state_preparations_per_call") != 1:
        raise ValueError("Official denominator must not have an extra norm contraction")
    repeats = data["measured_repeats"]
    if repeats < 3 or data.get("warmup_repeats") != 1 or data.get("threads") != 1:
        raise ValueError("Missing repeats, warmup or matched thread count")
    rows = data["rows"]
    expected = {(method, repeat) for method in methods for repeat in range(repeats + 1)}
    keys = [(row["method"], row["repeat"]) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unplanned timing rows")
    for row in rows:
        values = [row[k] for k in ("runtime_s", "value", "exact_value", "abs_error")]
        if not all(map(math.isfinite, values)) or row["runtime_s"] <= 0:
            raise ValueError("Nonfinite or nonpositive measurement")
        if abs(row["value"] - row["exact_value"]) > 1e-10 or row["abs_error"] > 1e-10:
            raise ValueError("Observable correctness failed")
    for method in methods[1:]:
        state = data["full_state"][method]
        if (not all(math.isfinite(state[k]) for k in ("norm_squared", "fidelity", "max_state_error")) or
                abs(state["norm_squared"] - 1) > 1e-10 or abs(state["fidelity"] - 1) > 1e-10 or
                state["max_state_error"] > 1e-9):
            raise ValueError("Independent full-state correctness failed")
    means = {method: statistics.mean(row["runtime_s"] for row in rows
             if row["method"] == method and row["repeat"] > 0) for method in methods}
    for method in methods:
        if not math.isclose(means[method], data["means_s"][method], rel_tol=1e-12):
            raise ValueError("Recorded means do not match raw timing rows")
    speedups = {method: means["upstream_typo_only"] / means[method] for method in methods[1:]}
    return {"means_s": means, "speedups_vs_upstream_typo_only": speedups,
            "rows_verified": len(rows), "full_state_methods_verified": len(methods) - 1}


def audit(directory, require_current=True):
    data = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    methods = ("upstream_typo_only", *data["full_state"])
    if methods not in (METHODS, ("upstream_typo_only", "mixed_schedule", "parity_insertions"),
                       ("upstream_typo_only", "prior_default", "integrated")):
        raise ValueError("Unrecognized predeclared comparison methods")
    result = validate_result(data, summary, methods)
    if hashlib.sha256((directory / "input.qasm").read_text(encoding="utf-8").encode()).hexdigest() != data["qasm_sha256"]:
        raise ValueError("QASM hash mismatch")
    for relative, digest in data["source_hashes"].items():
        path = ROOT / relative
        snapshot = directory / (path.stem + "_snapshot.py")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Snapshot mismatch: {relative}")
        if require_current and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Current source differs from measured source: {relative}")
    original = (directory / "upstream_quimb_original.py").read_text(encoding="utf-8")
    repaired = (directory / "upstream_quimb_typo_only.py").read_text(encoding="utf-8")
    from benchmarks.asc_revision_cpu.upstream_reference import source_pair, UPSTREAM_COMMIT
    if data["reference"]["commit"] != UPSTREAM_COMMIT or (original, repaired) != source_pair():
        raise ValueError("Reference is not the pinned official source and its sole typo repair")
    if original.count("optimize=self.contraction_optimizer") != 1 or repaired != original.replace(
            "optimize=self.contraction_optimizer", "optimize=self.contractions_optimizer"):
        raise ValueError("Official reference has changes beyond the one typo repair")
    for key, text in (("original_sha256", original), ("typo_only_sha256", repaired)):
        if hashlib.sha256(text.encode()).hexdigest() != data["reference"][key]:
            raise ValueError("Pinned reference source hash mismatch")
    from benchmarks.asc_revision_cpu.probe_linear_holdout import build
    circuit = build("mixed_entanglers", data["seed"], data["qubits"], data["layers"])
    if (hashlib.sha256(circuit.to_qasm().encode()).hexdigest() != data["qasm_sha256"] or
            len(circuit.queue) != data["gates"] or circuit.depth != data["depth"]):
        raise ValueError("Input cannot be reproduced from the recorded generator and seed")
    return {"recorded_at": datetime.now().astimezone().isoformat(), "status": "verified",
            "directory": str(directory.resolve()), **result,
            "qubits": data["qubits"], "gates": data["gates"], "depth": data["depth"],
            "source_snapshots_verified": len(data["source_hashes"]),
            "current_source_matches_required": require_current,
            "reference_extra_norm_contraction": False,
            "formal_30q_acceptance_complete": False, "all_requirements_complete": False,
            "scope": "one new mixed-gate input, not a universal or formal 30-qubit speed claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-only", action="store_true",
                        help="Verify immutable historical snapshots, not the current production checkout")
    args = parser.parse_args()
    report = audit(args.directory, require_current=not args.snapshot_only)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
