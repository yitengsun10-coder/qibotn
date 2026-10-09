"""Independent post-run audit. Partial/failed cases never yield a speed claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS=('upstream_typo_only','integrated','guarded_scaled_candidate')
KINDS=('chain','ring','nonlocal_mixed')
OFFICIAL_COMMIT='7ebbd4d4a00f37b898c552922953591b78b0af7e'
OFFICIAL_SHA='1e1ada931f9fb0f9228f7190cd46f5401328b1fbec9e42f40ba9dc7fca313aca'
REPAIRED_SHA='1a4e2e70c05e5015006ec2cdce3b81f6d2ba463ebbf604726fbb370fe4c9f472'


def audit(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    failures=[]
    def require(condition, explanation):
        if not condition: failures.append(explanation)
    verified=0
    for name,digest in summary['source_hashes'].items():
        path=root/'source_snapshot'/name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,
                'source snapshot mismatch: '+name)
        verified+=int(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest)
    require(summary.get('status')=='pass_supplementary_only','suite is not fully complete')
    require(summary.get('source_unchanged') is True,'source stability not confirmed')
    require(summary.get('original30_complete') is False,'original30 must remain incomplete')
    require(summary.get('all_contest_tasks_complete') is False,'whole contest not complete')
    require([c['kind'] for c in summary['cases']]==list(KINDS),'missing/reordered case summaries')
    require([c['kind'] for c in summary['workloads']]==list(KINDS),'missing/reordered input declarations')
    require(not (root/'external_drain_record.json').exists(),'suite externally interrupted; not complete')
    outputs=[]
    for spec in summary['workloads']:
        kind=spec['kind']
        path=root/kind/'result.json'
        if not path.exists():
            require(False,'missing case result: '+kind)
            continue
        before=len(failures)
        data=json.loads(path.read_text(encoding='utf-8'))
        require(data.get('status')=='pass',kind+': incomplete result')
        require(data.get('formal_supplementary_30q_complete') is True,kind+': no full acceptance')
        reference=data.get('reference',{})
        require(reference.get('commit')==OFFICIAL_COMMIT and
                reference.get('original_sha256')==OFFICIAL_SHA and
                reference.get('typo_only_sha256')==REPAIRED_SHA and
                reference.get('extra_norm_contraction') is False and
                reference.get('state_preparations_per_call')==1 and
                reference.get('algorithm_or_backend_replacement') is False,
                kind+': baseline identity/scope mismatch')
        original_path=root/kind/'upstream_quimb_original.py'
        repaired_path=root/kind/'upstream_quimb_typo_only.py'
        if original_path.exists() and repaired_path.exists():
            original,repaired=original_path.read_bytes(),repaired_path.read_bytes()
            old,new=b'optimize=self.contraction_optimizer',b'optimize=self.contractions_optimizer'
            require(hashlib.sha256(original).hexdigest()==OFFICIAL_SHA and
                    hashlib.sha256(repaired).hexdigest()==REPAIRED_SHA and
                    original.count(old)==1 and original.replace(old,new)==repaired,
                    kind+': official baseline bytes/sole typo fix changed')
        else: require(False,kind+': missing pinned official source files')
        encoded=(root/kind/'input.json').read_bytes()
        record=json.loads(encoded)
        require(hashlib.sha256(encoded).hexdigest()==spec['input_sha256']==data['input_sha256'],
                kind+': input bytes/hash mismatch')
        require(record.get('qubits')==30 and len(record.get('gates',[]))==data['gates']==spec['gates'],
                kind+': input scale mismatch')
        require(9500<=data['gates']<=10100 and 400<=data['depth']<=700,kind+': scale outside declaration')
        require(data['cutoff']==0 and data['max_bond'] is None and data['precision']=='complex128',
                kind+': precision/truncation mismatch')
        require(data.get('oracle_gates_applied')==data['gates'],kind+': oracle skipped gates')
        calibration=data.get('baseline_thread_calibration',[])
        require([r['threads'] for r in calibration]==[1,4,8],kind+': calibration incomplete')
        if len(calibration)==3:
            chosen=min(calibration,key=lambda r:r['runtime_s'])['threads']
            require(data.get('threads')==chosen,kind+': did not choose fastest official calibration')
            require(all(math.isfinite(r['runtime_s']+r['value']) and r['runtime_s']>0
                        and abs(r['value']-data['independent_observable'])<=1e-10 for r in calibration),
                    kind+': calibration correctness failed')
        pools=[p for p in data.get('threadpools',[]) if p['user_api']=='blas']
        require(bool(pools) and all(p['num_threads']==data.get('threads') for p in pools),
                kind+': BLAS threads differ')
        require(set(data.get('full_state',{}))==set(METHODS),kind+': missing independent states')
        for method, state in data.get('full_state',{}).items():
            norm0,norm1=state['oracle_norm_squared'],state['mps_norm_squared']
            overlap=complex(state['overlap_real'],state['overlap_imag'])
            if min(norm0,norm1)<=0 or not all(math.isfinite(x) for x in [norm0,norm1,overlap.real,overlap.imag]):
                require(False,kind+': zero/nonfinite full-state data: '+method)
                continue
            fidelity=abs(overlap)**2/(norm0*norm1)
            require(all(math.isfinite(x) for x in [norm0,norm1,fidelity]) and
                max(abs(norm0-1),abs(norm1-1),abs(fidelity-1))<=1e-9,
                kind+': full-state failure: '+method)
            require(state.get('full_state_comparison_pass') is True,kind+': full-state pass absent: '+method)
            require(state['gates_applied']==data['gates'] and state['oracle_branches']==64,
                    kind+': oracle scope mismatch: '+method)
        rows=data.get('rows',[])
        expected=[(m,r) for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]
        require([(r['method'],r['repeat']) for r in rows]==expected,kind+': missing/reordered planned rows')
        require(all(math.isfinite(r['runtime_s']+r['value']) and r['runtime_s']>0
            and abs(r['value']-data['independent_observable'])<=1e-10 for r in rows),kind+': observable mismatch')
        case_summary=next((c for c in summary['cases'] if c['kind']==kind),None)
        require(case_summary is not None and case_summary['exit_code']==0 and
                case_summary['stop_reason'] is None and case_summary['pass'],kind+': stop/exit was not clean')
        means={}
        if [(r['method'],r['repeat']) for r in rows]==expected:
            for method in METHODS:
                values=[r['runtime_s'] for r in rows if r['method']==method and r['repeat']>0]
                means[method]=statistics.mean(values)
                require(math.isclose(means[method],data['means_s'][method],rel_tol=1e-12),kind+': reported mean changed')
                require(math.isclose(statistics.stdev(values),data['stdev_s'][method],rel_tol=1e-12,abs_tol=1e-12),
                        kind+': reported deviation changed')
            require(all(math.isclose(means[METHODS[0]]/means[m],
                        data['speedups_vs_tuned_official'][m],rel_tol=1e-12) for m in METHODS[1:]),
                    kind+': reported speedup changed')
        if case_summary is not None:
            budgets=summary.get('budgets_per_case',{})
            require(case_summary.get('peak_process_tree_rss_bytes',math.inf)<=budgets.get('rss_mib',0)*2**20,
                    kind+': memory budget exceeded')
        outputs.append({'kind':kind,'pass':len(failures)==before,'rows':len(rows),
                        'means_s':means if len(failures)==before else None,
                        'speedups_vs_tuned_official':{m:means[METHODS[0]]/means[m] for m in METHODS[1:]}
                           if len(failures)==before else None})
    passed=not failures and len(outputs)==3
    aggregate=None
    if passed:
        aggregate={m:sum(c['means_s'][METHODS[0]] for c in outputs)/sum(c['means_s'][m] for c in outputs)
                   for m in METHODS[1:]}
    return {'status':'pass_supplementary_only' if passed else 'incomplete_or_failed',
            'supplement30_suite_complete':passed,'original30_complete':False,
            'all_contest_tasks_complete':False,'source_snapshots_verified':verified,
            'failures':failures,'cases':outputs,'aggregate_speedups':aggregate,
            'scope':'all three predeclared controlled-inter-region exact30 workloads only; not original high-entanglement30'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior audit; choose a new output')
    result=audit(args.root)
    result['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['supplement30_suite_complete'] else 1)
