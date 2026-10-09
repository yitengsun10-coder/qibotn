from benchmarks.asc_revision_cpu.probe_controlled_rank2 import build_input,QASM_SHA256
from benchmarks.asc_revision_cpu.supplement30_workloads import metadata,build,CONFIGURATIONS


def test_original30_input_is_not_replaced():
    circuit,encoded,spec=build_input('original30')
    assert (circuit.nqubits,len(circuit.queue),circuit.depth)==(30,9998,449)
    assert spec['input_sha256']==QASM_SHA256
    assert not spec['replaces_original30']


def test_supplement_inputs_identical_to_predeclared_v1():
    for kind,seed in CONFIGURATIONS:
        circuit,encoded,spec=build_input(kind)
        original=metadata(build(kind,seed),kind,seed)
        assert spec['input_sha256']==original['input_sha256']
        assert circuit.nqubits==30 and not spec['replaces_original30']


def test_smoke_inputs_are_labelled_by_actual_small_scale():
    for kind in ['original30','chain','ring','nonlocal_mixed']:
        _,_,spec=build_input(kind,True)
        assert spec['qubits']<30
