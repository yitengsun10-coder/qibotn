"""Explicit time-stopped chain continuation, not a new/selected benchmark.

Preserve the complete original suite byte-for-byte, its independent full
states and every completed row. Add three fresh-process warmups outside the
planned twelve rows, then execute only their missing suffix. Never promotes
the remaining ring/nonlocal cases or original high-entanglement30.
"""
import argparse
from copy import deepcopy
from contextlib import contextmanager,nullcontext
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from benchmarks.asc_revision_cpu.paired_supplement30_common_policy import METHODS,SOURCES,summarize_rows
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import ROOT,effective_memory,save

POLICY='other BLAS=1; QR width<512:1,<1024:4,<2048:8,else16; shared by ALL methods'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    return [(m,r) for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]


def validate_checkpoint(summary,data,smoke=False):
    cases=summary.get('cases',[])
    if (summary.get('status')!='incomplete_or_failed' or summary.get('source_unchanged') is not True or
            len(cases)!=1 or cases[0].get('kind')!='chain' or
            cases[0].get('stop_reason')!='wall_time_budget' or cases[0].get('exit_code')!=-9 or
            cases[0].get('pass') is not False):
        raise ValueError('Only a finalized, protected time-stop of the first chain case can resume')
    if (summary.get('smoke_only') is not smoke or data.get('smoke_only') is not smoke or
            data.get('original30_complete') is not False or data.get('all_contest_tasks_complete') is not False or
            data.get('formal_supplementary_30q_complete') is not False):
        raise ValueError('Unexpected scope or previously accepted result')
    if data.get('qubits')!=(9 if smoke else 30) or (not smoke and data.get('gates')!=9693):
        raise ValueError('Unexpected fixed chain input')
    if (data.get('precision')!='complex128' or data.get('cutoff')!=0 or data.get('max_bond') is not None or
            data.get('warmup_repeats')!=1 or data.get('measured_repeats')!=3 or
            data.get('common_thread_policy')!=POLICY or (not smoke and len(data.get('affinity',[]))!=16)):
        raise ValueError('Numerical/repetition/thread policy changed')
    if set(data.get('full_state',{}))!=set(METHODS) or data.get('oracle_gates_applied')!=data['gates']:
        raise ValueError('Three independent complete states required')
    for state in data['full_state'].values():
        a,b=state['oracle_norm_squared'],state['mps_norm_squared']
        z=complex(state['overlap_real'],state['overlap_imag'])
        if min(a,b)<=0 or not all(math.isfinite(x) for x in (a,b,z.real,z.imag)):
            raise ValueError('Invalid complete-state evidence')
        if (max(abs(a-1),abs(b-1),abs(abs(z)**2/(a*b)-1))>1e-9 or
                state.get('full_state_comparison_pass') is not True or
                state.get('gates_applied')!=data['gates'] or state.get('oracle_branches')!=64):
            raise ValueError('Complete-state check did not pass')
    pools=[p for p in data.get('outer_threadpools',[]) if p['user_api']=='blas']
    if not pools or any(p['num_threads']!=1 for p in pools):raise ValueError('Outer BLAS policy changed')
    rows=data['rows'];keys=[(r['method'],r['repeat']) for r in rows]
    if not rows or keys!=plan()[:len(rows)] or len(rows)>=len(plan()):
        raise ValueError('Completed rows must be a nonempty strict prefix')
    exact=data['independent_observable']
    if not math.isfinite(exact):raise ValueError('Missing independent scalar')
    for row in rows:
        validate_row(row,exact)
    return plan()[len(rows):]


def validate_row(row,exact):
    if (not all(math.isfinite(row[k]) for k in ('runtime_s','value','abs_error')) or row['runtime_s']<=0 or
            abs(row['value']-exact)>1e-10 or row['abs_error']>1e-10 or
            not math.isclose(row['abs_error'],abs(row['value']-exact),rel_tol=1e-9,abs_tol=1e-15)):
        raise ValueError('Invalid completed scalar/timing row')
    hist=row.get('qr_thread_call_counts',{})
    if set(hist)!={'1','4','8','16'} or any(not isinstance(v,int) or v<0 for v in hist.values()) or sum(hist.values())<=0:
        raise ValueError('Missing common QR scheduling evidence')


def check_sources(summary):
    expected={p.relative_to(ROOT).as_posix() for p in SOURCES}
    if set(summary['source_hashes'])!=expected:raise ValueError('Frozen fourteen-source inventory changed')
    for name,sha in summary['source_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen measured source changed: '+name)


@contextmanager
def observe_rank2_splits():
    """Unmeasured restart warmup only: preserve the registered mathematics.

    Circuit.psi copies the state and loses private counters. Observe the
    live state's counter increments instead, restoring the method on exit.
    This wrapper is never installed around any formal timed row.
    """
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS
    from functools import wraps
    original=ControlledRank2MPS._two_site
    evidence={'rank2_splits_observed':0,'trace_scope':'unmeasured_restart_warmup_only'}
    @wraps(original)
    def traced(state,*args,**kwargs):
        before=state.controlled_rank2_splits
        result=original(state,*args,**kwargs)
        evidence['rank2_splits_observed']+=state.controlled_rank2_splits-before
        return result
    ControlledRank2MPS._two_site=traced
    try:yield evidence
    finally:ControlledRank2MPS._two_site=original


def worker(args):
    import psutil
    from threadpoolctl import threadpool_info,threadpool_limits
    from qibotn.backends.quimb import QuimbNumpyBackend
    from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import make_backend
    from benchmarks.asc_revision_cpu.common_qr_thread_policy import common_qr_policy
    from benchmarks.asc_revision_cpu.supplement30_workloads import build,input_bytes
    from benchmarks.asc_revision_cpu.upstream_reference import load_reference
    from benchmarks.asc_large_cpu.benchmark import make_observable
    prior=args.output/'prior_segment'
    summary=json.loads((prior/'summary.json').read_text(encoding='utf-8'))
    original=json.loads((prior/'chain/result.json').read_text(encoding='utf-8'))
    remaining=validate_checkpoint(summary,original,args.smoke);check_sources(summary)
    data=json.loads((args.output/'chain/result.json').read_text(encoding='utf-8'))
    if digest(Path(__file__))!=data['continuation']['resume_script_sha256']:raise ValueError('Resume source changed')
    process=psutil.Process()
    if not set(data['affinity']).issubset(process.cpu_affinity()):raise ValueError('Original CPUs unavailable')
    process.cpu_affinity(data['affinity'])
    data['resumed_affinity']=process.cpu_affinity()
    data['resumed_versions']={name:importlib.metadata.version(name) for name in summary['environment']['versions']}
    if data['resumed_versions']!=summary['environment']['versions']:raise ValueError('Numerical dependency versions changed')
    circuit=build('chain',original['seed'],3 if args.smoke else 10,8 if args.smoke else 111)
    encoded=input_bytes(circuit)
    if encoded!=(prior/'chain/input.json').read_bytes() or hashlib.sha256(encoded).hexdigest()!=data['input_sha256']:
        raise ValueError('Regenerated input differs')
    cls,reference=load_reference(args.output/'chain')
    if reference!=original['reference']:raise ValueError('Official reference changed')
    methods={METHODS[0]:cls(),METHODS[1]:QuimbNumpyBackend(),METHODS[2]:make_backend()}
    for backend in methods.values():backend.configure_tn_simulation(max_bond_dimension=None,svd_cutoff=0)
    labels,sites,coefficients=make_observable(circuit.nqubits)
    target=args.output/'chain/result.json'
    with threadpool_limits(limits=1),common_qr_policy() as calls:
        data['resumed_outer_threadpools']=threadpool_info()
        if any(p['num_threads']!=1 for p in data['resumed_outer_threadpools'] if p['user_api']=='blas'):
            raise ValueError('Resumed outer BLAS differs')
        for method,repeat,warmup in [(m,-1,True) for m in METHODS]+[(m,r,False) for m,r in remaining]:
            data['status']=('restart_warmup_' if warmup else 'timing_')+method+'_'+str(repeat);save(target,data)
            before=dict(calls);begin=time.perf_counter()
            context=observe_rank2_splits() if warmup and method=='controlled_rank2' else nullcontext(None)
            with context as trace:
                value=float(methods[method].exp_value_observable_symbolic(circuit,labels,sites,coefficients,circuit.nqubits))
            row={'method':method,'repeat':repeat,'runtime_s':time.perf_counter()-begin,'value':value,
                 'abs_error':abs(value-data['independent_observable']),
                 'qr_thread_call_counts':{str(n):calls[n]-before[n] for n in calls},'segment':'explicit_chain_resume'}
            if trace is not None:row.update(trace)
            validate_row(row,data['independent_observable'])
            (data['restart_warmups'] if warmup else data['rows']).append(row);save(target,data)
            print(json.dumps(row),flush=True)
    check_sources(summary)
    if input_bytes(circuit)!=encoded or data['rows'][:len(original['rows'])]!=original['rows']:
        raise ValueError('Input or preserved rows changed')
    data.update(summarize_rows(data['rows']))
    data.update(status='pass_smoke_only' if args.smoke else 'pass',formal_supplementary_30q_complete=not args.smoke)
    save(target,data)


def suite(args):
    import psutil
    source=args.source.resolve()
    resolved_output=args.output.resolve()
    if resolved_output==source or source in resolved_output.parents:
        raise ValueError('Continuation output must not be inside the preserved source')
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    original=json.loads((source/'chain/result.json').read_text(encoding='utf-8'))
    remaining=validate_checkpoint(summary,original,args.smoke);check_sources(summary)
    if (source/'external_drain_record.json').exists():raise ValueError('Not a protected final time-stop')
    # No new worker while an old suite/worker with this exact script identity remains.
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if any(Path(arg).name=='paired_supplement30_common_policy.py' for arg in command if arg):
                raise ValueError('Original formal controller/worker still active')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps({'plan_only':not args.execute,'remaining_calls':remaining,
          'fresh_process_warmups':list(METHODS),'original30_complete':False,'all_contest_tasks_complete':False}),flush=True)
    if not args.execute:return
    if args.output.exists():raise FileExistsError('Fresh continuation directory required')
    if effective_memory(psutil.virtual_memory().available)['available_bytes']<(args.memory_mib+2048)*2**20:
        raise ValueError('Insufficient cgroup-aware headroom')
    args.output.mkdir(parents=True)
    shutil.copytree(source,args.output/'prior_segment')
    inventory={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    for name,sha in inventory.items():
        if digest(args.output/'prior_segment'/name)!=sha:raise ValueError('Prior bytes changed in backup')
    shutil.copytree(source/'source_snapshot',args.output/'source_snapshot')
    script_name=Path(__file__).relative_to(ROOT).as_posix()
    script_target=args.output/'source_snapshot'/script_name
    script_target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,script_target)
    target=args.output/'chain';target.mkdir()
    for name in ('input.json','upstream_quimb_original.py','upstream_quimb_typo_only.py'):
        shutil.copyfile(source/'chain'/name,target/name)
    data=deepcopy(original)
    data.update(status='resuming',restart_warmups=[],continuation={
        'prior_files_sha256':inventory,'preserved_row_count':len(original['rows']),'remaining_plan':remaining,
        'resume_script_sha256':digest(Path(__file__)),'original_exit_code':summary['cases'][0]['exit_code'],
        'original_stop_reason':summary['cases'][0]['stop_reason'],'fresh_process_warmups_per_method':1,
        'same_input_and_fourteen_frozen_sources':True,'scope':'chain only; ring/nonlocal/original30 remain incomplete'})
    save(target/'result.json',data)
    report=deepcopy(summary)
    report.update(status='running_chain_continuation',recorded_at=datetime.now().astimezone().isoformat(),cases=[],
                  source_unchanged=False,segmented_run_disclosed=True,
                  budgets_per_case={'seconds':args.seconds,'rss_mib':args.memory_mib,'reserve_mib':2048})
    report['source_hashes'][script_name]=digest(Path(__file__))
    save(args.output/'summary.json',report)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    command=[sys.executable,'-X','utf8',str(Path(__file__)),'--worker','--source',str(source),'--output',str(args.output)]
    if args.smoke:command.append('--smoke')
    begin=time.monotonic();peak=0;reason=None
    with (target/'run.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(args.output/'run_metadata.json',{'controller_pid':os.getpid(),'worker_pid':child.pid,
            'started_at':datetime.now().astimezone().isoformat(),'command':command})
        tree=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                children=tree.children(recursive=True);rss=sum(p.memory_info().rss for p in [tree,*children]);peak=max(peak,rss)
                if rss>args.memory_mib*2**20:reason='process_tree_memory_budget'
                elif effective_memory(psutil.virtual_memory().available)['available_bytes']<2048*2**20:reason='effective_memory_reserve'
                elif time.monotonic()-begin>args.seconds:reason='wall_time_budget'
                if reason:
                    for p in [*children,tree]:
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        code=child.wait()
    final=json.loads((target/'result.json').read_text(encoding='utf-8'))
    passed=code==0 and reason is None and final['status']==('pass_smoke_only' if args.smoke else 'pass')
    unchanged=all(digest(ROOT/name)==sha for name,sha in report['source_hashes'].items())
    report.update(status='complete_chain_only' if passed and unchanged else 'incomplete_or_failed',
        source_unchanged=unchanged,chain_continuation_complete=passed and unchanged and not args.smoke,
        supplement30_suite_complete=False,not_run_after_failure=['ring','nonlocal_mixed'],
        cases=[{'kind':'chain','exit_code':code,'stop_reason':reason,'wall_s':time.monotonic()-begin,
                'peak_process_tree_rss_bytes':peak,'pass':passed and unchanged}])
    save(args.output/'summary.json',report);print(json.dumps(report),flush=True)
    if not passed or not unchanged:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--seconds',type=float,default=7200)
    parser.add_argument('--memory-mib',type=float,default=12288)
    args=parser.parse_args()
    if not all(math.isfinite(v) and v>0 for v in (args.seconds,args.memory_mib)):parser.error('Finite positive budgets required')
    worker(args) if args.worker else suite(args)
