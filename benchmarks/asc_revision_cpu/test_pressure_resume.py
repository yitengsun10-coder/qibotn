from copy import deepcopy
import pytest
from benchmarks.asc_revision_cpu.resume_pressure_chain import validate_checkpoint, plan
from benchmarks.asc_revision_cpu.audit_pressure_chain import validate_continuation
from benchmarks.asc_revision_cpu.test_pressure_chain import valid


def stopped():
    data, _ = valid()
    lookup = {(r['method'], r['repeat']): r for r in data['rows']}
    data['rows'] = [lookup[key] for key in plan()[:3]]
    data['status'] = 'timing_upstream_typo_only'
    data['source_hashes'] = {'example.py': 'test-hash'}
    return data, {'status': 'incomplete_or_failed', 'exit_code': 1,
                  'stop_reason': 'wall_clock_budget_exceeded'}


def test_strict_prefix_can_resume():
    data, summary = stopped()
    remaining, exact = validate_checkpoint(data, summary)
    assert remaining == plan()[3:] and exact == .2


@pytest.mark.parametrize('kind', ['quality', 'missing_scalar', 'duplicate', 'out_of_order', 'different_precision', 'different_stop'])
def test_reject_bad_checkpoint(kind):
    data, summary = stopped()
    if kind == 'quality':
        data['full_state']['integrated']['fidelity'] = .28
    elif kind == 'missing_scalar':
        data['rows'][0]['exact_value'] = None
    elif kind == 'duplicate':
        data['rows'].append(deepcopy(data['rows'][-1]))
    elif kind == 'out_of_order':
        data['rows'].reverse()
    elif kind == 'different_precision':
        data['precision'] = 'complex64'
    elif kind == 'different_stop':
        summary['stop_reason'] = 'quality_failure'
    with pytest.raises(ValueError):
        validate_checkpoint(data, summary)


def continued():
    original, original_summary = stopped()
    data = deepcopy(original)
    base = data['rows'][0]
    data['rows'] += [{**base, 'method': method, 'repeat': repeat, 'segment': 'resumed_process'} for method, repeat in plan()[3:]]
    data['restart_warmups'] = [{**base, 'method': method, 'repeat': -1, 'segment': 'resumed_process'}
                               for method in ('upstream_typo_only', 'integrated')]
    data['resumed_threadpools'] = [{'num_threads': 1}]
    data['continuation'] = {'preserved_row_count': 3, 'remaining_plan': [list(k) for k in plan()[3:]],
        'original_exit_code': 1, 'original_stop_reason': 'wall_clock_budget_exceeded'}
    summary = {'segmented_run_disclosed': True, 'original_segment_exit_code': 1,
        'original_segment_stop_reason': 'wall_clock_budget_exceeded'}
    return data, summary, original, original_summary


def test_continuation_preserves_original_and_discloses_stop():
    report = validate_continuation(*continued())
    assert report['preserved_rows'] == 3 and report['resumed_rows'] == 5


@pytest.mark.parametrize('kind', ['prefix_changed', 'full_state_changed', 'missing_warmup', 'wrong_thread', 'hidden_stop'])
def test_reject_incorrect_continuation(kind):
    data, summary, original, original_summary = continued()
    if kind == 'prefix_changed':
        data['rows'][0]['runtime_s'] *= .5
    elif kind == 'full_state_changed':
        data['full_state']['integrated']['max_state_error'] *= 2
    elif kind == 'missing_warmup':
        data['restart_warmups'].pop()
    elif kind == 'wrong_thread':
        data['resumed_threadpools'][0]['num_threads'] = 20
    elif kind == 'hidden_stop':
        summary['original_segment_exit_code'] = 0
    with pytest.raises(ValueError):
        validate_continuation(data, summary, original, original_summary)
