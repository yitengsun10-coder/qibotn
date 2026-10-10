"""Compare shape forecasts against actual small-circuit MPS execution."""
import numpy as np
import pytest
from qibo import Circuit,gates
from qibo.backends import NumpyBackend

from benchmarks.asc_large_cpu.benchmark import Workload,build_circuit
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import (
    ControlledRank2MPS,make_backend,prepare,
)
from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import compile_mixed
from benchmarks.asc_revision_cpu.forecast_original30_shapes import forecast,original30_plan
from qibotn.backends import _exact_numpy as exact
from qibotn.backends import _structured_numpy as structured


@pytest.mark.parametrize('n,layers',[(6,4),(8,5),(10,8),(12,8),(14,12)])
@pytest.mark.parametrize('short',[False,True])
@pytest.mark.parametrize('rank2',[False,True])
def test_every_cnot_shape_matches_actual_execution(monkeypatch,n,layers,short,rank2):
    original=build_circuit(Workload(n,layers))
    circuit=compile_mixed(original)[0] if short else original
    predicted=forecast(circuit,rank2,True)
    observed=[]
    observed_array_bytes=[]
    live_splits=[]
    method=(ControlledRank2MPS._two_site if rank2 else structured.StructuredExactKernelMPS._two_site)
    def record(self,G,i,j,absorb,reverse=False):
        before=self.controlled_rank2_splits
        method(self,G,i,j,absorb,reverse=reverse)
        observed.append([1,*[self.bond_size(k,k+1) for k in range(self.L-1)],1])
        observed_array_bytes.append(sum(self[k].data.nbytes for k in range(self.L)))
        live_splits.append(self.controlled_rank2_splits-before)
    monkeypatch.setattr(ControlledRank2MPS,'_two_site',record)
    backend=make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0,cnot_optimization=False)
    state,_=prepare(backend,circuit)
    assert observed==[row['bonds'] for row in predicted['trace']]
    assert observed_array_bytes==[row['explicit_mps_array_bytes'] for row in predicted['trace']]
    assert max(observed_array_bytes)==predicted['max_explicit_mps_array_bytes']
    # Circuit.psi copies the state and resets this private counter. Observe
    # the live operation, not the already-documented copied-state field.
    assert predicted['controlled_rank2_dimension_splits']==sum(live_splits)
    assert live_splits==[int(row['rank2_dimension_split']) for row in predicted['trace']]
    # Independent numerical correctness remains separate from shape matching.
    dense_circuit=Circuit(n)
    dense_circuit.add(circuit.queue)
    oracle=NumpyBackend().execute_circuit(dense_circuit).state()
    np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1),oracle,atol=8e-11)


def test_reversed_adjacent_controls_match_actual_shapes(monkeypatch):
    circuit=Circuit(8)
    for layer in range(6):
        circuit.add(gates.RY(q,.13*(q+1)*(layer+1)) for q in range(8))
        circuit.add(gates.CNOT(q+1,q) for q in range(6,-1,-1))
    predicted=forecast(circuit,True,True)
    observed=[]
    method=ControlledRank2MPS._two_site
    def record(self,G,i,j,absorb,reverse=False):
        method(self,G,i,j,absorb,reverse=reverse)
        observed.append([1,*[self.bond_size(k,k+1) for k in range(self.L-1)],1])
    monkeypatch.setattr(ControlledRank2MPS,'_two_site',record)
    backend=make_backend()
    backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0,cnot_optimization=False)
    prepare(backend,circuit)
    assert observed==[row['bonds'] for row in predicted['trace']]


@pytest.mark.parametrize('gate',[gates.CNOT(0,3),gates.CZ(0,1),gates.TOFFOLI(0,1,2)])
def test_unsupported_input_fails_closed(gate):
    circuit=Circuit(4);circuit.add(gate)
    with pytest.raises(ValueError):forecast(circuit)


def test_original30_forecast_does_not_simulate_or_claim_acceptance(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No MPS preparation is allowed in this planning test')
    monkeypatch.setattr(exact,'prepare_routed_mps',forbidden)
    report=original30_plan()
    assert report['simulation_performed'] is False
    assert report['formal_30q_acceptance_complete'] is False
    assert report['publishable_speedup'] is None
    assert report['short_schedule_changed_blocks']==112
    for row in report['methods'].values():
        assert row['cnot_gates']==112*29
        assert row['one_site_gates']==30+112*60
        assert row['max_bond']<=32768
        assert row['observed_memory'] is False
        assert row['max_explicit_mps_array_bytes']==45812984448
        assert row['max_generic_merged_matrix_bytes']==17179869184
        event=row['max_generic_merge_plus_prior_state_event']
        assert row['max_generic_merge_plus_prior_state_bytes']==(
            event['prior_state_array_bytes']+event['merged_matrix_bytes'])
        assert 0<=event['queue_index']<9998
        assert 'not measured RSS' in row['combined_component_scope']
