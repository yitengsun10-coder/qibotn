import hashlib
from collections import Counter

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit
from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import (
    compile_mixed, make_backend, prepare, schedule_short_block,
)
from benchmarks.asc_revision_cpu.evaluate import build_circuit as build_general
from benchmarks.asc_revision_cpu.evaluate import dense_expectation, observable
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import QASM_SHA256
from qibotn.backends._linear_cnot import linear_rows


@pytest.mark.parametrize("count", [0, 1, 7, 8, 29, 32, 256, 257])
def test_bounds_operator_gate_multiset_and_precedence(count):
    rng = np.random.default_rng(431 + count)
    pairs = tuple(tuple(int(q) for q in rng.choice(10, 2, replace=False)) for _ in range(count))
    selected, cost, details = schedule_short_block(10, pairs, list(range(10)), 9)
    assert Counter(selected) == Counter(pairs)
    assert linear_rows(10, selected) == linear_rows(10, pairs)
    assert details["searched"] == (8 <= count <= 256)
    if details["searched"]:
        assert details["noncommuting_precedence_verified"]
        assert cost[0] <= details["estimated_score_before"]
    else:
        assert selected == pairs


def test_original30_plan_only_input_unchanged_all_blocks_checked():
    circuit = build_circuit(Workload(30, 112))
    original = circuit.to_qasm().encode("utf-8")
    assert hashlib.sha256(original).hexdigest() == QASM_SHA256
    compiled, details = compile_mixed(circuit)
    assert circuit.to_qasm().encode("utf-8") == original
    assert len(compiled.queue) == len(circuit.queue) == 9998
    blocks = details["short_cnot_schedule"]
    assert len(blocks) == 112
    assert all(b["gates"] == 29 and b["complete_gf2_operator_verified"] for b in blocks)
    assert any(b["changed"] for b in blocks)
    assert not details["short_schedule_uses_amplitudes_or_outputs"]


@pytest.mark.parametrize("qubits,layers", [(8, 4), (10, 4), (10, 8)])
def test_scaled_original_complete_state_and_scalar(qubits, layers):
    circuit = build_circuit(Workload(qubits, layers))
    original = circuit.to_qasm()
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, details = prepare(backend, circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=4e-11)
    assert details["production_integrated"] is False
    labels, sites, coefficients = observable(qubits, "general")
    expected = dense_expectation(oracle, qubits, labels, sites, coefficients)
    actual = backend.exp_value_observable_symbolic(circuit, labels, sites, coefficients, qubits)
    assert actual == pytest.approx(expected, abs=4e-11)
    assert circuit.to_qasm() == original


@pytest.mark.parametrize("kind", ["chain", "ring", "long_range", "all_to_all"])
@pytest.mark.parametrize("seed", [417, 919])
def test_general_connectivity_complete_state(kind, seed):
    circuit = build_general(kind, 8, 5, seed)
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, _ = prepare(backend, circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=4e-11)


def test_unsupported_gate_and_truncation_fallback():
    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.TOFFOLI(0, 2, 4))
    circuit.add(gates.RY(3, .31))
    backend = make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None, svd_cutoff=0)
    state, _ = prepare(backend, circuit)
    oracle = NumpyBackend().execute_circuit(circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=2e-11)
    backend.configure_tn_simulation(max_bond_dimension=8, svd_cutoff=1e-10)
    state, details = prepare(backend, circuit)
    assert details["engine"] == "native"
    assert details["linear_compiler"] == {"enabled": False}
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), oracle, atol=2e-11)


def test_probe_snapshot_dispatch_and_failed_admission_never_launches(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from benchmarks.asc_revision_cpu import probe_short_cnot_rank2 as probe
    assert probe.suite.__globals__["__file__"] == probe.__file__
    assert probe.HERE / "candidate_short_cnot_rank2.py" in probe.SOURCES
    assert probe.HERE / "common_qr_thread_policy.py" in probe.SOURCES
    def forbidden_launch(*args, **kwargs):
        raise AssertionError("Worker must not launch after failed memory admission")
    monkeypatch.setattr(probe.previous.subprocess, "Popen", forbidden_launch)
    args = SimpleNamespace(output=tmp_path / "blocked", kind="original30", smoke=False,
        candidate="short", threads=16, seconds=180., memory_mib=1e12,
        reserve_mib=2048., execute=True)
    with pytest.raises(ValueError, match="Memory admission failed"):
        probe.suite(args)
    report = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert report["input"]["input_sha256"] == QASM_SHA256
    assert report["formal_30q_acceptance_complete"] is False
    assert report["publishable_speedup"] is None
    assert not (args.output / "run.log").exists()
    for name, digest in report["source_hashes"].items():
        assert hashlib.sha256((args.output / "source_snapshot" / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("candidate,seconds,memory,reserve", [
    ("unknown", 180., 32768., 2048.), ("short", 301., 32768., 2048.),
    ("plain", 0., 32768., 2048.), ("short", 180., 65536., 2048.),
    ("short", 180., 32768., 0.),
])
def test_launch_rejects_scope_or_budget_expansion(tmp_path, candidate, seconds, memory, reserve):
    from benchmarks.asc_revision_cpu.launch_original30_probe import launch
    with pytest.raises(ValueError):
        launch(candidate, tmp_path / "must_not_launch", seconds, memory, reserve)


def test_launch_rejects_output_outside_results(tmp_path):
    from benchmarks.asc_revision_cpu.launch_original30_probe import launch
    with pytest.raises(ValueError, match="inside this repository"):
        launch("short", tmp_path / "outside")


def test_launch_rejects_undeclared_thread_policy(tmp_path):
    from benchmarks.asc_revision_cpu.launch_original30_probe import launch
    with pytest.raises(ValueError, match="declared common"):
        launch("short", tmp_path / "outside", policy="arbitrary")
