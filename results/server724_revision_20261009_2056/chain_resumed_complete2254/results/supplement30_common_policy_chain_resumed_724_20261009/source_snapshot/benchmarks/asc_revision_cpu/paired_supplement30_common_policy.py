"""Fresh exact30 supplementary pairs under an IDENTICAL QR thread policy.

Official pinned math is unchanged (one attribute typo repair only). All paths
receive the same size-only QR scheduling and one-thread other BLAS work.
Original high-entanglement30 remains incomplete, regardless of these results.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import CORE,ROOT,effective_memory,save

HERE=Path(__file__).resolve().parent
METHODS=('upstream_typo_only','integrated','controlled_rank2')
SOURCES=[*CORE,Path(__file__),*[HERE/name for name in (
    'common_qr_thread_policy.py','candidate_controlled_rank2.py','supplement30_workloads.py',
    'upstream_reference.py','profile_fixed30_bounded.py','paired_supplement30.py')],
    ROOT/'benchmarks/asc_large_cpu/benchmark.py']


def summarize_rows(rows):
    expected=[(m,r) for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]
    if [(r['method'],r['repeat']) for r in rows]!=expected:raise ValueError('Incomplete/reordered timings')
    if any(not math.isfinite(r['runtime_s']+r['abs_error']) or r['runtime_s']<=0 or r['abs_error']>1e-10 for r in rows):
        raise ValueError('Invalid timing/correctness')
    samples={m:[r['runtime_s'] for r in rows if r['method']==m and r['repeat']>0] for m in METHODS}
    means={m:statistics.mean(v) for m,v in samples.items()}
    return {'means_s':means,'stdev_s':{m:statistics.stdev(v) for m,v in samples.items()},
            'speedups_vs_official_common_qr_policy':{m:means[METHODS[0]]/means[m] for m in METHODS[1:]},
            'rank2_speedup_vs_integrated_common_qr_policy':means['integrated']/means['controlled_rank2']}


def worker(args):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits,threadpool_info
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import make_backend,prepare
    from benchmarks.asc_revision_cpu.common_qr_thread_policy import common_qr_policy
    from benchmarks.asc_revision_cpu.supplement30_workloads import CONFIGURATIONS,BlockSumOracle,build,input_bytes
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_revision_cpu.paired_supplement30 import check_state
    from benchmarks.asc_large_cpu.benchmark import make_observable
    declaration=json.loads((args.output.parent/'summary.json').read_text(encoding='utf-8'))
    for name,digest in declaration['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed')
    process=psutil.Process();allowed=process.cpu_affinity()
    if len(allowed)<16 and not args.smoke:raise ValueError('Predeclared sixteen-CPU resource cap unavailable')
    process.cpu_affinity(allowed[:16])
    size,layers=(3,8) if args.smoke else (10,111)
    circuit=build(args.kind,dict(CONFIGURATIONS)[args.kind],size,layers)
    encoded=input_bytes(circuit)
    spec=next(c for c in declaration['workloads'] if c['kind']==args.kind)
    if hashlib.sha256(encoded).hexdigest()!=spec['input_sha256']:raise ValueError('Input changed')
    (args.output/'input.json').write_bytes(encoded)
    reference_class,reference=load_reference(args.output)
    methods={METHODS[0]:reference_class(),METHODS[1]:QuimbNumpyBackend(),METHODS[2]:make_backend()}
    for backend in methods.values():backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    result={**spec,'status':'independent_oracle','reference':reference,'rows':[],'full_state':{},
            'smoke_only':args.smoke,'affinity':process.cpu_affinity(),'warmup_repeats':1,'measured_repeats':3,
            'formal_supplementary_30q_complete':False,'original30_complete':False,'all_contest_tasks_complete':False,
            'common_thread_policy':'other BLAS=1; QR width<512:1,<1024:4,<2048:8,else16; shared by ALL methods',
            'timing_scope':'new conversion/compiler + every gate + original four ZZ; oracle/imports/separate state checks excluded'}
    path=args.output/'result.json';save(path,result)
    labels,sites,coefficients=make_observable(circuit.nqubits)
    with threadpool_limits(limits=1),common_qr_policy() as qr_calls:
        result['outer_threadpools']=threadpool_info()
        oracle=BlockSumOracle.from_circuit(circuit,size)
        norm=oracle.expectation().real
        expected=sum(c*oracle.expectation(word,where).real for word,where,c in zip(labels,sites,coefficients))/norm
        result.update(independent_observable=expected,oracle_norm_squared=norm,oracle_gates_applied=oracle.gates_applied)
        save(path,result)
        for method,backend in methods.items():
            result['status']='full_state_'+method;save(path,result)
            if method==METHODS[0]:
                state=backend._qibo_circuit_to_quimb(circuit,quimb_circuit_type=backend.circuit_ansatz,
                    gate_opts={'max_bond':None,'cutoff':0}).psi
                details={'engine':'pinned_official_math_unchanged'}
            elif method==METHODS[1]:state,details=backend._prepare_optimized_mps(circuit)
            else:state,details=prepare(backend,circuit)
            if any(t.data.dtype!=np.complex128 for t in state.tensor_map.values()):raise ValueError('Unexpected state dtype')
            overlap,evidence=oracle.overlap_mps(state)
            result['full_state'][method]={**check_state(norm,evidence['mps_norm_squared'],overlap),**evidence,
                'overlap_real':overlap.real,'overlap_imag':overlap.imag,'max_bond_observed':state.max_bond(),'preparation':details}
            save(path,result)
            print(json.dumps({'kind':args.kind,'full_state_pass':method,'max_bond':state.max_bond()}),flush=True)
            del state;gc.collect()
        del oracle;gc.collect()
        for repeat in range(4):
            for method in (METHODS if repeat%2==0 else METHODS[::-1]):
                result['status']=f'timing_{method}_{repeat}';save(path,result)
                before=dict(qr_calls)
                begin=time.perf_counter()
                value=float(methods[method].exp_value_observable_symbolic(circuit,labels,sites,coefficients,circuit.nqubits))
                row={'method':method,'repeat':repeat,'runtime_s':time.perf_counter()-begin,
                     'value':value,'abs_error':abs(value-expected),
                     'qr_thread_call_counts':{str(n):qr_calls[n]-before[n] for n in qr_calls}}
                result['rows'].append(row);save(path,result)
                print(json.dumps({'kind':args.kind,**row}),flush=True)
                if not math.isfinite(value) or row['abs_error']>1e-10:raise ValueError('Observable mismatch')
        if input_bytes(circuit)!=encoded:raise ValueError('Input mutated')
        result.update(summarize_rows(result['rows']))
        result.update(status='pass_smoke_only' if args.smoke else 'pass',
                      formal_supplementary_30q_complete=not args.smoke)
        save(path,result)


def suite(args):
    import psutil
    from benchmarks.asc_revision_cpu.supplement30_workloads import CONFIGURATIONS,build,metadata
    if args.output.exists():raise ValueError('Fresh output required')
    args.output.mkdir(parents=True)
    size,layers=(3,8) if args.smoke else (10,111)
    report={'recorded_at':datetime.now().astimezone().isoformat(),'status':'plan_only','smoke_only':args.smoke,
            'execute_requested':args.execute,'original30_complete':False,'all_contest_tasks_complete':False,
            'supplement30_suite_complete':False,
            'source_hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
            'workloads':[metadata(build(k,s,size,layers),k,s,size) for k,s in CONFIGURATIONS],
            'common_policy':'same size-only QR policy for native official, integrated, rank2; other BLAS=1; affinity first16',
            'budgets_per_case':{'seconds':args.seconds,'rss_mib':args.memory_mib,'reserve_mib':2048},
            'environment':{'python':sys.version,'platform':platform.platform(),
                'memory':effective_memory(psutil.virtual_memory().available),'affinity':psutil.Process().cpu_affinity(),
                'versions':{n:importlib.metadata.version(n) for n in ['qibo','quimb','numpy','scipy','threadpoolctl','psutil']}},
            'cases':[]}
    for source in SOURCES:
        target=args.output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    save(args.output/'summary.json',report)
    if not args.execute:return
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    report['status']='running';save(args.output/'summary.json',report)
    for index,(kind,_) in enumerate(CONFIGURATIONS):
        if effective_memory(psutil.virtual_memory().available)['available_bytes']<(args.memory_mib+2048)*2**20:
            raise ValueError('Insufficient effective memory; no worker launched')
        target=args.output/kind;target.mkdir()
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker','--kind',kind,'--output',str(target)]
        if args.smoke:command.append('--smoke')
        begin=time.monotonic();peak=0;reason=None
        with (target/'run.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            tree=psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    children=tree.children(recursive=True)
                    rss=sum(p.memory_info().rss for p in [tree,*children]);peak=max(peak,rss)
                    if rss>args.memory_mib*2**20:reason='process_tree_memory_budget'
                    elif effective_memory(psutil.virtual_memory().available)['available_bytes']<2048*2**20:reason='effective_memory_reserve'
                    elif time.monotonic()-begin>args.seconds:reason='wall_time_budget'
                    if reason:
                        for p in [*children,tree]:
                            try:p.kill()
                            except psutil.NoSuchProcess:pass
                        break
                except psutil.NoSuchProcess:pass
                time.sleep(.1)
            code=process.wait()
        data=json.loads((target/'result.json').read_text(encoding='utf-8')) if (target/'result.json').exists() else {}
        passed=code==0 and reason is None and data.get('status')==('pass_smoke_only' if args.smoke else 'pass')
        report['cases'].append({'kind':kind,'exit_code':code,'stop_reason':reason,'wall_s':time.monotonic()-begin,
                               'peak_process_tree_rss_bytes':peak,'pass':passed})
        save(args.output/'summary.json',report);print(json.dumps(report['cases'][-1]),flush=True)
        if not passed:
            report['not_run_after_failure']=[k for k,_ in CONFIGURATIONS[index+1:]]
            break
    unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==d for n,d in report['source_hashes'].items())
    passed=unchanged and len(report['cases'])==3 and all(c['pass'] for c in report['cases'])
    report.update(status=('pass_smoke_only' if args.smoke else 'pass_supplementary_only') if passed else 'incomplete_or_failed',
                  source_unchanged=unchanged,supplement30_suite_complete=passed and not args.smoke)
    save(args.output/'summary.json',report)
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--kind',choices=['chain','ring','nonlocal_mixed'])
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--memory-mib',type=float,default=12288)
    args=parser.parse_args()
    if not all(math.isfinite(v) and v>0 for v in [args.seconds,args.memory_mib]):parser.error('Finite positive budgets required')
    if args.worker and args.kind is None:parser.error('Worker requires kind')
    worker(args) if args.worker else suite(args)
