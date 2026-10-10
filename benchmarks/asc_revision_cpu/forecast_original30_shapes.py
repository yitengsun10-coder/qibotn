"""Allocation-free shape forecast for adjacent CNOT + one-site unitary MPS.

This follows economy-QR output dimensions and the controlled-rank2 dimension
guard, NOT numerical Schmidt ranks or amplitudes. It is a resource forecast,
not a simulation, measured memory/runtime, fidelity check or acceptance result.
Unsupported gates/connectivity fail closed. No 2**30 array is ever allocated.
"""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.asc_large_cpu.benchmark import Workload, build_circuit
from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import compile_mixed
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import QASM_SHA256


def forecast(circuit, controlled_rank2=True, keep_trace=False):
    n = circuit.nqubits
    bonds = [1] * (n + 1)
    center = 0
    result = {'max_bond': 1, 'max_explicit_mps_array_bytes': 2*n*16,
        'max_generic_merged_matrix_bytes': 0, 'max_tall_qr_matrix_bytes': 0,
        'max_generic_merge_plus_prior_state_bytes': 0,
        'max_generic_merge_plus_prior_state_event': None,
        'first_max_bond_queue_index': None,
        'first_max_explicit_state_queue_index': None,
        'controlled_rank2_dimension_splits': 0, 'cnot_gates': 0,
        'one_site_gates': 0, 'trace': []}
    def state_bytes():
        return sum(2*bonds[k]*bonds[k+1]*16 for k in range(n))
    def account():
        arrays = state_bytes()
        if arrays > result['max_explicit_mps_array_bytes']:
            result['first_max_explicit_state_queue_index'] = index
        if max(bonds) > result['max_bond']:
            result['first_max_bond_queue_index'] = index
        result['max_explicit_mps_array_bytes'] = max(result['max_explicit_mps_array_bytes'],arrays)
        result['max_bond'] = max(result['max_bond'],max(bonds))
    def qr(rows, columns):
        if rows > columns:
            result['max_tall_qr_matrix_bytes'] = max(
                result['max_tall_qr_matrix_bytes'], rows*columns*16)
        return min(rows,columns)
    for index,gate in enumerate(circuit.queue):
        if len(gate.qubits)==1 and gate.name in {'h','rx','ry'}:
            result['one_site_gates'] += 1
            continue
        if gate.name not in {'cx','cnot'} or len(gate.qubits)!=2:
            raise ValueError('Forecast supports only H/RX/RY and adjacent CNOT')
        control,target = gate.qubits
        i,j = sorted((control,target))
        if j != i+1:
            raise ValueError('Nonlocal routing is outside this shape forecast')
        # Product-state initial centre choice changes only trivial rank-one
        # work. Subsequent exact routed CNOTs explicitly set centre=target.
        if center < i:
            for site in range(center,i):
                bonds[site+1] = qr(2*bonds[site],bonds[site+1])
                account()
        elif center > j:
            for site in range(center,j,-1):
                bonds[site] = qr(2*bonds[site+1],bonds[site])
                account()
        left,old,right = bonds[i],bonds[i+1],bonds[j+1]
        rank2 = controlled_rank2 and 2*old < min(2*left,2*right)
        if rank2:
            new = qr(2*(left if control<target else right),2*old)
            result['controlled_rank2_dimension_splits'] += 1
        else:
            merged_bytes = 4*left*right*16
            result['max_generic_merged_matrix_bytes'] = max(
                result['max_generic_merged_matrix_bytes'],merged_bytes)
            # Pair components at the SAME operation, not unrelated maxima.
            # This is shape accounting only: no allocator/alias/workspace
            # model, therefore not an RSS lower/upper bound or admission.
            combined = state_bytes()+merged_bytes
            if combined > result['max_generic_merge_plus_prior_state_bytes']:
                result['max_generic_merge_plus_prior_state_bytes'] = combined
                result['max_generic_merge_plus_prior_state_event'] = {
                    'queue_index':index, 'pair':[control,target],
                    'prior_state_array_bytes':state_bytes(),
                    'merged_matrix_bytes':merged_bytes}
            new = qr(2*left,2*right) if control<target else qr(2*right,2*left)
        bonds[i+1] = new
        center = target
        result['cnot_gates'] += 1
        account()
        if keep_trace:
            result['trace'].append({'original_queue_index':index,'pair':[control,target],
                'bonds':list(bonds),'rank2_dimension_split':bool(rank2),
                'explicit_mps_array_bytes':state_bytes()})
    result.update(final_bonds=bonds, final_explicit_mps_array_bytes=
        sum(2*bonds[k]*bonds[k+1]*16 for k in range(n)),
        controlled_rank2_enabled=controlled_rank2, qubits=n,
        simulation_performed=False, observed_memory=False, runtime_prediction_s=None,
        formal_30q_acceptance_complete=False, publishable_speedup=None,
        scope='dimension forecast only; not numerical rank, fidelity, measured allocation or time',
        combined_component_scope='same-operation shape components, not measured RSS or an allocation bound',
        excluded_memory=['identity gauge snapshots','QR private copies and workspace',
            'simultaneously live old/new tensors','gate/observable temporaries','Python/dependencies'])
    if not keep_trace:
        del result['trace']
    return result


def original30_plan():
    circuit=build_circuit(Workload(30,112))
    digest=hashlib.sha256(circuit.to_qasm().encode('utf-8')).hexdigest()
    if (digest,len(circuit.queue),circuit.depth)!=(QASM_SHA256,9998,449):
        raise ValueError('Original30 input changed')
    short,compiler=compile_mixed(circuit)
    return {'status':'shape_forecast_not_acceptance','input_sha256':digest,
        'qubits':30,'gates':9998,'depth':449,'precision':'complex128',
        'max_bond':None,'cutoff':0,'simulation_performed':False,
        'formal_30q_acceptance_complete':False,'publishable_speedup':None,
        'methods':{'generic_exact':forecast(circuit,False),
                   'controlled_rank2':forecast(circuit,True),
                   'short_cnot_controlled_rank2':forecast(short,True)},
        'short_schedule_changed_blocks':sum(b.get('changed',False)
            for b in compiler['short_cnot_schedule']),
        'note':'No allocation admission or runtime guarantee; validate shapes on actual smaller circuits first'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=original30_plan()
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
