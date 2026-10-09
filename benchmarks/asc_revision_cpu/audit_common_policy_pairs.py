"""Independent audit of complete common-policy pairs; never promotes original30."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from benchmarks.asc_revision_cpu.audit_supplement30 import OFFICIAL_SHA,REPAIRED_SHA,OFFICIAL_COMMIT

METHODS=('upstream_typo_only','integrated','controlled_rank2')
KINDS=('chain','ring','nonlocal_mixed')


def audit(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    failures=[];outputs=[];verified=0
    def require(condition,message):
        if not condition:failures.append(message)
    for name,digest in summary['source_hashes'].items():
        path=root/'source_snapshot'/name
        valid=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        verified+=bool(valid);require(valid,'source snapshot mismatch: '+name)
    require(summary.get('original30_complete') is False,'original30 cannot be promoted')
    require(summary.get('all_contest_tasks_complete') is False,'whole contest cannot be promoted')
    require(summary.get('smoke_only') is False,'smoke suite is not formal30')
    require(summary.get('source_unchanged') is True,'source stability unconfirmed')
    require(summary.get('status')=='pass_supplementary_only','suite incomplete')
    require([s['kind'] for s in summary['workloads']]==list(KINDS),'input declarations missing/reordered')
    require([s['kind'] for s in summary['cases']]==list(KINDS),'case summaries missing/reordered')
    trusted_sources=verified==len(summary['source_hashes']) and summary.get('source_unchanged') is True and summary.get('smoke_only') is False
    for spec in summary['workloads']:
        kind=spec['kind'];before=len(failures);path=root/kind/'result.json'
        if not path.exists():require(False,kind+': no result');continue
        data=json.loads(path.read_text(encoding='utf-8'))
        require(data.get('status')=='pass' and data.get('formal_supplementary_30q_complete') is True,
                kind+': not a complete formal case')
        require(data.get('smoke_only') is False and data.get('original30_complete') is False,
                kind+': scope mismatch')
        input_path=root/kind/'input.json'
        if not input_path.exists():require(False,kind+': missing input');continue
        encoded=input_path.read_bytes();record=json.loads(encoded)
        require(hashlib.sha256(encoded).hexdigest()==data['input_sha256']==spec['input_sha256'],kind+': input changed')
        require(record.get('qubits')==data['qubits']==30 and len(record.get('gates',[]))==data['gates']==spec['gates'],
                kind+': input scale changed')
        require(9500<=data['gates']<=10100 and 400<=data['depth']<=700 and spec['bridge_gate_count']==6,
                kind+': pressure declaration changed')
        require(data['precision']=='complex128' and data['cutoff']==0 and data['max_bond'] is None,
                kind+': precision or truncation changed')
        require(data['oracle_gates_applied']==data['gates'],kind+': skipped oracle gates')
        ref=data['reference']
        original=(root/kind/'upstream_quimb_original.py').read_bytes()
        repaired=(root/kind/'upstream_quimb_typo_only.py').read_bytes()
        old,new=b'optimize=self.contraction_optimizer',b'optimize=self.contractions_optimizer'
        require(hashlib.sha256(original).hexdigest()==ref['original_sha256']==OFFICIAL_SHA and
                hashlib.sha256(repaired).hexdigest()==ref['typo_only_sha256']==REPAIRED_SHA and
                original.count(old)==1 and original.replace(old,new)==repaired and ref['commit']==OFFICIAL_COMMIT,
                kind+': official source/sole typo fix changed')
        require(ref['extra_norm_contraction'] is False and ref['state_preparations_per_call']==1 and
                ref['algorithm_or_backend_replacement'] is False,kind+': reference scope changed')
        pools=[p for p in data.get('outer_threadpools',[]) if p['user_api']=='blas']
        require(bool(pools) and all(p['num_threads']==1 for p in pools) and len(data['affinity'])==16,
                kind+': outer threads/affinity changed')
        require(data.get('common_thread_policy')=='other BLAS=1; QR width<512:1,<1024:4,<2048:8,else16; shared by ALL methods',
                kind+': common policy declaration changed')
        require(set(data['full_state'])==set(METHODS),kind+': missing complete states')
        for method,state in data['full_state'].items():
            a,b=state['oracle_norm_squared'],state['mps_norm_squared']
            overlap=complex(state['overlap_real'],state['overlap_imag'])
            valid=min(a,b)>0 and all(math.isfinite(v) for v in [a,b,overlap.real,overlap.imag])
            fidelity=abs(overlap)**2/(a*b) if valid else math.inf
            require(valid and max(abs(a-1),abs(b-1),abs(fidelity-1))<=1e-9 and
                    state['full_state_comparison_pass'] is True and state['oracle_branches']==64 and
                    state['gates_applied']==data['gates'],kind+': independent state failure: '+method)
        direct=data['full_state'].get('controlled_rank2',{}).get('preparation',{}).get('controlled_rank2_splits',0)>0
        # The old Circuit.psi copy loses private counters. A disclosed
        # continuation may corroborate execution in its separately recorded,
        # unmeasured restart warmup; no old full-state field is rewritten.
        trace_rows=[r for r in data.get('restart_warmups',[]) if r.get('method')=='controlled_rank2' and r.get('repeat')==-1]
        script='benchmarks/asc_revision_cpu/resume_common_policy_chain.py'
        traced=(len(trace_rows)==1 and trace_rows[0].get('rank2_splits_observed',0)>0 and
                trace_rows[0].get('trace_scope')=='unmeasured_restart_warmup_only' and
                summary.get('segmented_run_disclosed') is True and
                summary['source_hashes'].get(script) is not None and
                summary['source_hashes'][script]==data.get('continuation',{}).get('resume_script_sha256'))
        require(direct or traced,kind+': source candidate execution counter absent (state-copy counter loss is not proof)')
        rows=data['rows'];expected=[(m,r) for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]
        complete=[(r['method'],r['repeat']) for r in rows]==expected
        require(complete,kind+': incomplete/reordered 12 planned rows')
        require(all(math.isfinite(r['value']+r['runtime_s']) and r['runtime_s']>0 and
                    abs(r['value']-data['independent_observable'])<=1e-10 for r in rows),kind+': scalar mismatch')
        require(all(set(r['qr_thread_call_counts'])=={'1','4','8','16'} and
                    all(isinstance(v,int) and v>=0 for v in r['qr_thread_call_counts'].values()) and
                    sum(r['qr_thread_call_counts'].values())>0 for r in rows),kind+': QR scheduling record invalid')
        means={};deviations={}
        if complete:
            for method in METHODS:
                sample=[r['runtime_s'] for r in rows if r['method']==method and r['repeat']>0]
                means[method]=statistics.mean(sample);deviations[method]=statistics.stdev(sample)
                require(math.isclose(means[method],data['means_s'][method],rel_tol=1e-12) and
                        math.isclose(deviations[method],data['stdev_s'][method],rel_tol=1e-12,abs_tol=1e-12),
                        kind+': reported statistics changed')
            require(all(math.isclose(means[METHODS[0]]/means[m],
                        data['speedups_vs_official_common_qr_policy'][m],rel_tol=1e-12) for m in METHODS[1:]),
                    kind+': reported speed changed')
        case=next((s for s in summary['cases'] if s['kind']==kind),{})
        require(case.get('pass') is True and case.get('exit_code')==0 and case.get('stop_reason') is None,
                kind+': stop/exit not clean')
        require(case.get('peak_process_tree_rss_bytes',math.inf)<=summary['budgets_per_case']['rss_mib']*2**20,
                kind+': memory budget exceeded')
        passed=len(failures)==before and trusted_sources
        outputs.append({'kind':kind,'pass':passed,'rows':len(rows),'means_s':means if passed else None,
                        'stdev_s':deviations if passed else None,
                        'speedups_vs_official_common_qr_policy':{m:means[METHODS[0]]/means[m] for m in METHODS[1:]}
                           if passed else None})
    passed=not failures and len(outputs)==3
    aggregate={m:sum(s['means_s'][METHODS[0]] for s in outputs)/sum(s['means_s'][m] for s in outputs)
               for m in METHODS[1:]} if passed else None
    return {'status':'pass_supplementary_only' if passed else 'incomplete_or_failed',
            'supplement30_suite_complete':passed,'original30_complete':False,'all_contest_tasks_complete':False,
            'source_snapshots_verified':verified,'cases':outputs,'failures':failures,
            'aggregate_speedups_vs_official_common_qr_policy':aggregate,
            'scope':'three predeclared six-bridge exact30 inputs; not generic high-entanglement original30'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh audit output required')
    result=audit(args.root)
    result['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['supplement30_suite_complete'] else 1)
