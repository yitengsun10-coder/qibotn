"""Accept only complete, reproducible and correct pressure evidence."""
from copy import deepcopy
import pytest
from benchmarks.asc_revision_cpu.audit_pressure_chain import validate_pressure


def valid():
    state = {'norm_squared': 1., 'fidelity': 1., 'max_state_error': 1e-15}
    data = {'status': 'pass', 'calibration_only': False, 'qubits': 20, 'layers': 169,
        'gates': 9991, 'depth': 677, 'precision': 'complex128', 'cutoff': 0, 'max_bond': None,
        'threads': 1, 'warmup_repeats': 1, 'measured_repeats': 3,
        'threadpools': [{'num_threads': 1}],
        'reference': {'extra_norm_contraction': False, 'state_preparations_per_call': 1},
        'full_state': {key: deepcopy(state) for key in ('upstream_typo_only', 'integrated')},
        'rows': [{'method': method, 'repeat': repeat, 'runtime_s': runtime, 'value': .2,
                  'exact_value': .2, 'abs_error': 0.}
                 for repeat in range(4) for method, runtime in [('upstream_typo_only', 10.), ('integrated', 4.)]],
        'means_s': {'upstream_typo_only': 10., 'integrated': 4.}, 'speedup': 2.5}
    summary = {'status': 'pass', 'exit_code': 0, 'stop_reason': None, 'calibration_only': False}
    return data, summary


def test_predeclared_size_is_not_tutorial():
    from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit
    circuit = build_circuit(Workload(20, 169))
    assert len(circuit.queue) == 9991 and circuit.depth == 677
    assert len({q for g in circuit.queue for q in g.qubits}) == 20
    assert {g.name for g in circuit.queue} == {'h', 'rx', 'ry', 'cx'}


def test_valid_pressure():
    data, summary = valid()
    report = validate_pressure(data, summary)
    assert report['speedup'] == 2.5 and report['rows_verified'] == 8
    assert report['full_state_methods_verified'] == 2


@pytest.mark.parametrize('kind', ['calibration', 'reduced_input', 'partial', 'stopped',
    'bad_reference_state', 'missing_reference_state', 'wrong_threads', 'wrong_speed', 'lower_precision'])
def test_reject_invalid_pressure(kind):
    data, summary = valid()
    if kind == 'calibration':
        data['calibration_only'] = True
    elif kind == 'reduced_input':
        data['gates'] = 100
    elif kind == 'partial':
        data['rows'].pop()
    elif kind == 'stopped':
        summary['stop_reason'] = 'wall_clock_budget_exceeded'
    elif kind == 'bad_reference_state':
        data['full_state']['upstream_typo_only']['fidelity'] = .28
    elif kind == 'missing_reference_state':
        del data['full_state']['upstream_typo_only']
    elif kind == 'wrong_threads':
        data['threadpools'][0]['num_threads'] = 20
    elif kind == 'wrong_speed':
        data['speedup'] = 7.
    elif kind == 'lower_precision':
        data['precision'] = 'complex64'
    with pytest.raises(ValueError):
        validate_pressure(data, summary)
