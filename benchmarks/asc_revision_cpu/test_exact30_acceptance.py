"""Small helper tests are not a completed 30q simulation or speed result."""
from copy import deepcopy
import math
from types import SimpleNamespace
import pytest
from benchmarks.asc_revision_cpu.exact30_acceptance import (
    METHODS, container_resource_plan, export_state, make_plan, streamed_overlap,
    summarize_rows, validate_state)


def rows():
    return [{'method': method, 'repeat': repeat, 'runtime_s': 10. if method == METHODS[0] else 2., 'value': .2}
            for repeat in range(4) for method in (METHODS if repeat % 2 == 0 else METHODS[::-1])]


def test_plan_does_not_admit_local_16_gib():
    plan = make_plan(16 * 2**30, 6 * 2**30, 200 * 2**30, 4 * 2**30)
    assert not plan['resources_admitted']
    assert not plan['formal_30q_acceptance_complete']
    assert plan['publishable_speedup'] is None
    assert (plan['qubits'], plan['gates'], plan['depth']) == (30, 9998, 449)
    assert plan['cutoff'] == 0 and plan['max_bond'] is None


def test_plan_can_admit_larger_cpu_without_claiming_pass():
    plan = make_plan(128 * 2**30, 120 * 2**30, 200 * 2**30, 96 * 2**30)
    assert plan['resources_admitted'] and not plan['formal_30q_acceptance_complete']


def test_plan_rejects_insufficient_disk():
    assert not make_plan(128 * 2**30, 120 * 2**30, 1 * 2**30, 96 * 2**30)['resources_admitted']


def test_plan_rejects_60_gib_container_on_large_host(tmp_path):
    (tmp_path / 'memory.max').write_text(str(60 * 2**30))
    (tmp_path / 'memory.current').write_text(str(2 * 2**30))
    memory = SimpleNamespace(total=377 * 2**30, available=329 * 2**30)
    plan = container_resource_plan(memory, 50 * 2**30, 96 * 2**30, tmp_path)
    assert not plan['resources_admitted']
    assert plan['host_total_memory_bytes'] == 377 * 2**30
    assert plan['effective_total_memory_bytes'] == 60 * 2**30
    assert plan['effective_available_memory_bytes'] == 58 * 2**30
    assert not plan['formal_30q_acceptance_complete']
    assert not plan['allocation_forecast_complete']


def test_plan_uses_host_memory_when_no_container_limit(tmp_path):
    memory = SimpleNamespace(total=128 * 2**30, available=120 * 2**30)
    plan = container_resource_plan(memory, 200 * 2**30, 96 * 2**30, tmp_path)
    assert plan['resources_admitted']
    assert plan['effective_available_memory_bytes'] == memory.available
    assert plan['container_memory']['cgroup_limit_bytes'] is None
    assert not plan['allocation_forecast_complete']


def test_complete_fair_rows():
    report = summarize_rows(rows())
    assert report['speedup'] == 5.
    assert report['means_s'] == {METHODS[0]: 10., METHODS[1]: 2.}


@pytest.mark.parametrize('kind', ['partial', 'duplicate', 'reorder', 'wrong_value', 'nan_value', 'zero_time', 'nan_time'])
def test_reject_invalid_rows(kind):
    data = deepcopy(rows())
    if kind == 'partial':
        data.pop()
    elif kind == 'duplicate':
        data.append(deepcopy(data[-1]))
    elif kind == 'reorder':
        data[0], data[1] = data[1], data[0]
    elif kind == 'wrong_value':
        data[-1]['value'] += 1e-3
    elif kind == 'nan_value':
        data[-1]['value'] = math.nan
    elif kind == 'zero_time':
        data[-1]['runtime_s'] = 0.
    elif kind == 'nan_time':
        data[-1]['runtime_s'] = math.nan
    with pytest.raises(ValueError):
        summarize_rows(data)


def test_global_phase_is_not_a_state_error():
    report = validate_state(1., 1., complex(0, 1))
    assert report['full_state_comparison_pass'] and not report['independent_dense_30q_oracle']


@pytest.mark.parametrize('values', [(1., .28, .28), (1., 1., .8), (1., 0., 0.), (1., 1., math.nan)])
def test_reject_bad_state(values):
    with pytest.raises(ValueError):
        validate_state(*values)


def test_streamed_full_overlap_matches_independent_dense_contraction(tmp_path):
    import numpy as np
    import quimb.tensor as qtn
    rng = np.random.default_rng(20261009)
    left = rng.normal(size=32) + 1j*rng.normal(size=32)
    right = rng.normal(size=32) + 1j*rng.normal(size=32)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    reference = qtn.MatrixProductState.from_dense(left, dims=[2]*5, cutoff=0, max_bond=None)
    actual = qtn.MatrixProductState.from_dense(right, dims=[2]*5, cutoff=0, max_bond=None)
    export_state(reference, tmp_path / 'reference', np)
    overlap = streamed_overlap(tmp_path / 'reference', actual, np)
    assert abs(overlap - np.vdot(left, right)) < 1e-12
    same = streamed_overlap(tmp_path / 'reference', reference, np)
    assert abs(same - 1) < 1e-12
    validate_state(1., 1., same)


def test_modified_reference_tensor_is_rejected(tmp_path):
    import numpy as np
    import quimb.tensor as qtn
    vector = np.array([1., 0, 0, 0], dtype=np.complex128)
    state = qtn.MatrixProductState.from_dense(vector, dims=[2, 2], cutoff=0, max_bond=None)
    destination = tmp_path / 'reference'
    export_state(state, destination, np)
    path = destination / 'site_00.npy'
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='bytes changed'):
        streamed_overlap(destination, state, np)
