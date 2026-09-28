"""Verify committed ASC evidence without rerunning the long benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "evidence" / "formal_thread_sweep_q30" / "thread_sweep_summary.json"
DEFAULT20 = ROOT / "evidence" / "formal_default20_q30"
FIDELITY = ROOT / "evidence" / "fidelity_scan_q30.json"
PROVENANCE = ROOT / "evidence" / "provenance.json"


def main() -> None:
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    default20 = json.loads((DEFAULT20 / "summary.json").read_text(encoding="utf-8"))
    fidelity = json.loads(FIDELITY.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert formal["qubits"] == 30
    assert formal["layers"] == 112
    assert formal["repeats"] == 3
    assert formal["fixed_numerical_configuration"] == {
        "cutoff": 1e-12,
        "max_bond": 128,
        "variant": "baseline",
    }
    by_threads = {row["threads"]: row for row in formal["results"]}
    one, sixteen = by_threads[1], by_threads[16]
    assert one["qasm_sha256"] == sixteen["qasm_sha256"]
    assert sixteen["correct_within_tolerance"]
    assert sixteen["abs_error_vs_1_thread"] <= formal["tolerance"]

    for threads in (1, 16):
        csv_path = (
            FORMAL.parent / f"threads_{threads}" / "results.csv"
        )
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == formal["repeats"]
        assert {int(row["qubits"]) for row in rows} == {30}
        assert {int(row["gates"]) for row in rows} == {9_998}
        assert {int(row["depth"]) for row in rows} == {449}
        assert {row["qasm_sha256"] for row in rows} == {
            one["qasm_sha256"]
        }
        assert {float(row["cutoff"]) for row in rows} == {1e-12}
        assert {int(row["max_bond"]) for row in rows} == {128}
        assert {int(row["threads"]) for row in rows} == {threads}

    speedup = sixteen["steady_mean_runtime_s"] / one["steady_mean_runtime_s"]
    assert speedup > 1.0

    default_values = default20["q30_l112"]["variants"]["baseline"]
    with (DEFAULT20 / "results.csv").open(newline="", encoding="utf-8") as stream:
        default_rows = list(csv.DictReader(stream))
    assert len(default_rows) == 3
    assert {int(row["qubits"]) for row in default_rows} == {30}
    assert {int(row["gates"]) for row in default_rows} == {9_998}
    assert {int(row["depth"]) for row in default_rows} == {449}
    assert {int(row["threads"]) for row in default_rows} == {20}
    assert {float(row["cutoff"]) for row in default_rows} == {1e-12}
    assert {int(row["max_bond"]) for row in default_rows} == {128}
    assert {row["qasm_sha256"] for row in default_rows} == {
        one["qasm_sha256"]
    }
    default_error = abs(default_values["mean_expectation"] - one["expectation"])
    assert default_error <= formal["tolerance"]
    default_speedup = (
        default_values["steady_mean_runtime_s"]
        / one["steady_mean_runtime_s"]
    )
    assert default_speedup > speedup
    assert provenance["default_thread_probe"]["num_threads"] == 20
    assert provenance["official_upstream_base_commit"].startswith("7ebbd4d")

    assert fidelity["qubits"] == 30
    assert fidelity["gates"] >= 9_000
    assert fidelity["depth"] >= 400
    assert fidelity["variants"]["baseline"]["fidelity_pass"]
    rejected = [
        row for name, row in fidelity["variants"].items() if name != "baseline"
    ]
    assert rejected and all(not row["fidelity_pass"] for row in rejected)

    print("Evidence verification: PASS")
    print(
        f"Workload: {fidelity['qubits']} qubits, "
        f"{fidelity['gates']} gates, depth {fidelity['depth']}"
    )
    print(f"Scaling 16-thread -> 1-thread speedup: {speedup:.4f}x")
    print(f"Formal default 20-thread -> 1-thread speedup: {default_speedup:.4f}x")
    print(f"Formal absolute output difference: {default_error:.3e}")


if __name__ == "__main__":
    main()
