"""Bounded source-candidate diagnosis on unchanged original/supplementary inputs.

Instrumented calls are not formal timings. Even a complete call does not
complete the original30 acceptance. Never compute a ratio from partial paths.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmarks.asc_revision_cpu.profile_fixed30_bounded import CORE, ROOT, QASM_SHA256, Recorder, ProbeStop, effective_memory, save

HERE=Path(__file__).resolve().parent
SOURCES=[*CORE,Path(__file__),*[HERE/name for name in (
    'candidate_controlled_rank2.py','candidate_controlled_adaptive.py','profile_fixed30_bounded.py','supplement30_workloads.py',
    'paired_supplement30.py','upstream_reference.py')],ROOT/'benchmarks/asc_large_cpu/benchmark.py']


def build_input(kind,smoke=False):
    from benchmarks.asc_large_cpu.benchmark import Workload,build_circuit
    from benchmarks.asc_revision_cpu.supplement30_workloads import CONFIGURATIONS,build,input_bytes,metadata
    if kind=='original30':
        circuit=build_circuit(Workload(6,3) if smoke else Workload(30,112))
        encoded=circuit.to_qasm().encode('utf-8')
        digest=hashlib.sha256(encoded).hexdigest()
        if not smoke and (len(circuit.queue),circuit.depth,digest)!=(9998,449,QASM_SHA256):
            raise ValueError('Original input changed')
        spec={'kind':kind,'qubits':circuit.nqubits,'gates':len(circuit.queue),'depth':circuit.depth,
              'input_sha256':digest,'input_file':'input.qasm','replaces_original30':False}
    else:
        group_size,layers=(3,8) if smoke else (10,111)
        seed=dict(CONFIGURATIONS)[kind]
        circuit=build(kind,seed,group_size,layers)
        encoded=input_bytes(circuit)
        spec={**metadata(circuit,kind,seed,group_size),'input_file':'input.json'}
    return circuit,encoded,spec


def worker(args):
    import numpy as np
    import psutil
    import quimb.tensor as qtn
    from qibo.backends import NumpyBackend
    from threadpoolctl import threadpool_limits,threadpool_info
    from qibotn.backends import _exact_numpy as exact
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import make_backend,ControlledRank2MPS
    from benchmarks.asc_large_cpu.benchmark import make_observable
    from benchmarks.asc_revision_cpu.paired_supplement30 import check_state
    from benchmarks.asc_revision_cpu.supplement30_workloads import BlockSumOracle
    declaration=json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
    for name,digest in declaration['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed after predeclaration')
    process=psutil.Process()
    allowed=process.cpu_affinity()
    if args.threads>len(allowed):raise ValueError('Threads exceed allowed CPUs')
    process.cpu_affinity(allowed[:max(4,args.threads)])
    circuit,encoded,spec=build_input(args.kind,args.smoke)
    if spec!=declaration['input']:raise ValueError('Predeclared input changed')
    (args.output/spec['input_file']).write_bytes(encoded)
    state_class=ControlledRank2MPS
    if args.candidate=='adaptive':
        from benchmarks.asc_revision_cpu.candidate_controlled_adaptive import make_backend,AdaptiveControlledMPS
        state_class=AdaptiveControlledMPS
    backend=make_backend();backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    labels,sites,coefficients=make_observable(circuit.nqubits)
    result={**spec,'status':'running','formal_30q_acceptance_complete':False,'publishable_speedup':None,
            'instrumented_not_formal_timing':True,'precision':'complex128','cutoff':0,'max_bond':None,
            'smoke_test_only':args.smoke,'candidate':args.candidate,'affinity':process.cpu_affinity(),'threads':args.threads,
            'max_bond_observed':1,'controlled_rank2_splits':0,'completed_apply_gate_calls':0}
    begin=time.perf_counter();last_write=0.;last_state=None
    def guard(extra):
        if process.memory_info().rss+extra>args.memory_mib*2**20:
            raise ProbeStop('process_rss_or_projected_workspace_budget')
        if effective_memory(psutil.virtual_memory().available)['available_bytes']-extra<args.reserve_mib*2**20:
            raise ProbeStop('effective_memory_or_projected_workspace_reserve')
        if time.perf_counter()-begin>args.seconds:raise ProbeStop('worker_time_budget')
    def observe(force=False):
        nonlocal last_write
        now=time.perf_counter()
        if not force and now-last_write<1.:return
        result.update(instrumented_elapsed_s=now-begin,stages=recorder.stages,
                      in_flight_stages=[frame[0] for frame in recorder.stack],
                      memory=effective_memory(psutil.virtual_memory().available),
                      current_process_rss_bytes=process.memory_info().rss)
        save(args.output/'profile.json',result);last_write=now
    def after_gate(arguments,keywords):
        nonlocal last_state
        last_state=arguments[0]._psi
        result['max_bond_observed']=max(result['max_bond_observed'],last_state.max_bond())
        result['controlled_rank2_splits']=getattr(last_state,'controlled_rank2_splits',0)
        result['completed_apply_gate_calls']+=1
    def matrix_metadata(arguments,keywords):
        matrix=arguments[0]
        return {'input_bytes':matrix.nbytes,'shape':list(matrix.shape),'estimated_extra_bytes':4*matrix.nbytes}
    recorder=Recorder(guard,observe)
    with threadpool_limits(limits=args.threads):
        result['threadpools']=threadpool_info()
        if any(p['num_threads']!=args.threads for p in result['threadpools'] if p['user_api']=='blas'):
            raise ValueError('BLAS limit ineffective')
        try:
            recorder.patch(qtn.CircuitMPS,'apply_gate','apply_gate_inclusive',after=after_gate)
            recorder.patch(qtn.CircuitMPS,'apply_gate_raw','apply_gate_raw_inclusive',after=after_gate)
            recorder.patch(state_class,'_two_site','controlled_rank2.two_site')
            recorder.patch(exact,'_left_isometric_factor','exact.isometric_factor',before=matrix_metadata)
            recorder.patch(exact,'_reduced_qr','exact.reduced_qr',before=matrix_metadata)
            observe(True)
            value=float(backend.exp_value_observable_symbolic(circuit,labels,sites,coefficients,circuit.nqubits))
            if not math.isfinite(value):raise ValueError('Nonfinite observable')
            result.update(status='completed_instrumented_call_not_acceptance',observable=value,
                          instrumented_call_s=time.perf_counter()-begin)
        except ProbeStop as error:
            result.update(status='bounded_incomplete',stop_reason=str(error))
        finally:
            recorder.restore();observe(True)
        if result['status']=='completed_instrumented_call_not_acceptance':
            if args.kind!='original30':
                result['status']='independent_full_state_running_not_acceptance';observe(True)
                with threadpool_limits(limits=1):
                    oracle=BlockSumOracle.from_circuit(circuit,3 if args.smoke else 10)
                    overlap,evidence=oracle.overlap_mps(last_state)
                    norm=oracle.expectation().real
                    result['independent_full_state']={**check_state(norm,evidence['mps_norm_squared'],overlap),**evidence}
                result['status']='completed_instrumented_call_not_acceptance'
            elif args.smoke:
                oracle=NumpyBackend().execute_circuit(circuit).state()
                vector=np.asarray(last_state.to_dense()).reshape(-1)
                result['smoke_max_abs_error']=float(np.max(np.abs(vector-oracle)))
                if result['smoke_max_abs_error']>1e-10:raise ValueError('Smoke state mismatch')
            observe(True)
    return 0 if result['status']=='completed_instrumented_call_not_acceptance' else 2


def suite(args):
    import psutil
    if args.output.exists():raise ValueError('Fresh output required')
    args.output.mkdir(parents=True)
    _,_,spec=build_input(args.kind,args.smoke)
    report={'recorded_at':datetime.now().astimezone().isoformat(),'status':'plan_only',
            'execute_requested':args.execute,'input':spec,'threads':args.threads,'candidate':args.candidate,
            'budgets':{'seconds':args.seconds,'memory_mib':args.memory_mib,'reserve_mib':args.reserve_mib},
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
            'formal_30q_acceptance_complete':False,'publishable_speedup':None}
    for source in SOURCES:
        target=args.output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    save(args.output/'summary.json',report)
    if not args.execute:return
    if effective_memory(psutil.virtual_memory().available)['available_bytes']<(args.memory_mib+args.reserve_mib)*2**20:
        raise ValueError('Memory admission failed; no worker launched')
    command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',
             '--kind',args.kind,'--output',str(args.output),'--threads',str(args.threads),
             '--candidate',args.candidate,
             '--seconds',str(args.seconds),'--memory-mib',str(args.memory_mib),
             '--reserve-mib',str(args.reserve_mib)]
    if args.smoke:command.append('--smoke')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    begin=time.monotonic();peak=0;reason=None
    report['status']='running';save(args.output/'summary.json',report)
    with (args.output/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        tree=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                children=tree.children(recursive=True)
                rss=sum(p.memory_info().rss for p in [tree,*children]);peak=max(peak,rss)
                if rss>args.memory_mib*2**20:reason='supervisor_process_tree_memory_budget'
                elif effective_memory(psutil.virtual_memory().available)['available_bytes']<args.reserve_mib*2**20:
                    reason='supervisor_effective_memory_reserve'
                elif time.monotonic()-begin>args.seconds:reason='supervisor_wall_time_budget'
                if reason:
                    for p in [*children,tree]:
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.1)
        code=process.wait()
    report.update(status='complete_diagnostic_only' if code==0 and reason is None else 'bounded_or_failed',
                  exit_code=code,supervisor_stop_reason=reason,wall_s=time.monotonic()-begin,
                  peak_process_tree_rss_bytes=peak,
                  source_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                                       for name,digest in report['source_hashes'].items()))
    save(args.output/'summary.json',report);print(json.dumps(report,indent=2))
    if code!=0 or reason or not report['source_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['original30','chain','ring','nonlocal_mixed'],required=True)
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--candidate',choices=['plain','adaptive'],default='plain')
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--memory-mib',type=float,default=12288)
    parser.add_argument('--reserve-mib',type=float,default=2048)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.threads<1 or not all(math.isfinite(v) and v>0 for v in [args.seconds,args.memory_mib,args.reserve_mib]):
        parser.error('Finite positive budgets/threads required')
    raise SystemExit(worker(args) if args.worker else suite(args))
