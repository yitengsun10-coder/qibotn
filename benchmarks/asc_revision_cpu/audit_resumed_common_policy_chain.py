"""Audit explicit chain continuation; cannot promote the full suite/original30."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
from benchmarks.asc_revision_cpu.audit_common_policy_pairs import audit as audit_pairs
from benchmarks.asc_revision_cpu.resume_common_policy_chain import METHODS,SOURCES,ROOT,validate_checkpoint,validate_row


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    data=json.loads((root/'chain/result.json').read_text(encoding='utf-8'))
    prior=root/'prior_segment'
    original_summary=json.loads((prior/'summary.json').read_text(encoding='utf-8'))
    original=json.loads((prior/'chain/result.json').read_text(encoding='utf-8'))
    errors=[]
    def require(condition,message):
        if not condition:errors.append(message)
    try:remaining=validate_checkpoint(original_summary,original)
    except (ValueError,KeyError,TypeError) as exc:
        errors.append('Original checkpoint invalid: '+str(exc));remaining=[]
    inventory=data.get('continuation',{}).get('prior_files_sha256',{})
    actual_files={p.relative_to(prior).as_posix() for p in prior.rglob('*') if p.is_file()}
    require(set(inventory)==actual_files and bool(inventory),'Original segment file inventory differs')
    for name,sha in inventory.items():
        path=PurePosixPath(name)
        safe=not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name
        require(safe and (prior/name).is_file() and digest(prior/name)==sha,'Prior raw bytes differ: '+name)
    expected={p.relative_to(ROOT).as_posix() for p in SOURCES}
    require(set(original_summary['source_hashes'])==expected,'Original fourteen-source declaration differs')
    script='benchmarks/asc_revision_cpu/resume_common_policy_chain.py'
    require(set(summary['source_hashes'])==expected|{script},'Continuation source inventory differs')
    for name,sha in original_summary['source_hashes'].items():
        require(summary['source_hashes'].get(name)==sha,'Measured source declaration changed: '+name)
    require(summary['source_hashes'].get(script)==data['continuation'].get('resume_script_sha256'),
            'Resume script identity differs')
    require(summary['workloads']==original_summary['workloads'],'Predeclared inputs changed')
    require(data['continuation'].get('preserved_row_count')==len(original['rows']) and
            data['continuation'].get('remaining_plan')==[list(x) for x in remaining], 'Continuation plan changed')
    require(data['rows'][:len(original['rows'])]==original['rows'],'Completed original rows changed')
    for key in ('full_state','reference','independent_observable','input_sha256','common_thread_policy','affinity',
                'precision','cutoff','max_bond','gates','qubits','depth','outer_threadpools'):
        require(data.get(key)==original.get(key),'Preserved evidence changed: '+key)
    require((root/'chain/input.json').read_bytes()==(prior/'chain/input.json').read_bytes(),'Input raw bytes changed')
    warmups=data.get('restart_warmups',[])
    require([(r.get('method'),r.get('repeat')) for r in warmups]==[(m,-1) for m in METHODS],
            'Missing/reordered fresh-process warmups')
    for row in warmups:
        try:validate_row(row,data['independent_observable'])
        except (ValueError,KeyError,TypeError) as exc:errors.append('Bad restart warmup: '+str(exc))
    require(data.get('resumed_affinity')==original.get('affinity'),'Resumed CPU affinity differs')
    require(data.get('resumed_versions')==original_summary['environment']['versions'],'Numerical dependencies changed')
    pools=[p for p in data.get('resumed_outer_threadpools',[]) if p['user_api']=='blas']
    require(bool(pools) and all(p['num_threads']==1 for p in pools),'Resumed outer threads differ')
    require(data['continuation'].get('original_exit_code')==-9 and
            data['continuation'].get('original_stop_reason')=='wall_time_budget' and
            summary.get('segmented_run_disclosed') is True,'Original protected stop not disclosed')
    require(summary.get('status')=='complete_chain_only' and summary.get('chain_continuation_complete') is True,
            'Continuation is not complete')
    require(summary.get('supplement30_suite_complete') is False and summary.get('original30_complete') is False and
            summary.get('all_contest_tasks_complete') is False,'Incomplete broader scope promoted')
    # Existing independent auditor checks all hashes, official raw bytes,
    # input, complete states, twelve rows, statistics, common policy and exit.
    # Missing ring/nonlocal results intentionally keep the whole suite false.
    paired=audit_pairs(root)
    chain=next((c for c in paired['cases'] if c['kind']=='chain'),{})
    require(chain.get('pass') is True,'Independent chain pair audit did not pass')
    require(paired['supplement30_suite_complete'] is False,'Incomplete three-case suite promoted')
    passed=not errors
    return {'status':'pass_explicit_chain_continuation_only' if passed else 'incomplete_or_failed',
            'formal_chain_supplement_complete':passed,'supplement30_suite_complete':False,
            'original30_complete':False,'all_contest_tasks_complete':False,
            'failures':errors,'prior_raw_files_verified':len(inventory),
            'preserved_rows':len(original['rows']),'restart_warmups':len(warmups),'total_planned_rows':len(data['rows']),
            'chain':chain if passed else None,'broader_pair_audit':paired,
            'scope':'one predeclared six-bridge chain, split across disclosed time-stopped and resumed processes'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh audit output required')
    result=audit(args.root);result['auditor_sha256']=digest(Path(__file__))
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['formal_chain_supplement_complete'] else 1)
