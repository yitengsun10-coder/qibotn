"""Preserve and drain this owned suite after its first 15-minute calibration failure.

Does not alter the suite's original summary, result, exit codes or snapshots.
The external stop is a separate record, never a successful completion. No
other process is eligible. The plan is read-only unless --execute is given.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
EXPECTED=ROOT/'results/supplement30_exact_paired_724_20261009'


def eligible(summary):
    cases=summary.get('cases',[])
    return bool(cases and cases[0]['kind']=='chain' and not cases[0]['pass']
                and cases[0]['stop_reason']=='wall_time_budget'
                and cases[0]['wall_s']>=900)


def main(execute=False):
    import psutil
    summary_path=EXPECTED/'summary.json'
    content=summary_path.read_bytes()
    summary=json.loads(content)
    if not eligible(summary):
        raise ValueError('First calibration has not recorded its bounded failure; do not stop')
    matches=[]
    for process in psutil.process_iter(['pid','name','cmdline','create_time']):
        command=process.info['cmdline'] or []
        if ('python' in (process.info['name'] or '').lower() and '--worker' not in command
            and any(Path(arg).name=='paired_supplement30.py' for arg in command)
            and '--output' in command
            and (ROOT/command[command.index('--output')+1]).resolve()==EXPECTED.resolve()
            and Path(process.cwd()).resolve()==ROOT.resolve()):
            matches.append(process)
    if len(matches)!=1:
        raise ValueError(f'Expected one owned controller, found {len(matches)}')
    controller=matches[0]
    processes=[*controller.children(recursive=True),controller]
    identities=[{'pid':p.pid,'created_at':p.create_time(),'command':p.cmdline()} for p in processes]
    report={'recorded_at':datetime.now().astimezone().isoformat(),
            'reason':'first complete official calibration exceeded 900s; defer further suite calls for bounded rank2 diagnosis',
            'summary_sha256_before_stop':hashlib.sha256(content).hexdigest(),
            'original_summary_preserved':True,'planned_processes':identities,
            'execute_requested':execute,'supplement30_suite_complete':False,
            'original30_complete':False,'all_contest_tasks_complete':False}
    if not execute:
        print(json.dumps(report,indent=2));return
    record=EXPECTED/'external_drain_record.json'
    if record.exists():raise ValueError('Existing stop record must not be overwritten')
    record.write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Freeze the controller first so it cannot launch another worker during drain.
    for p,identity in zip(processes,identities):
        if p.create_time()!=identity['created_at'] or p.cmdline()!=identity['command']:
            raise ValueError('Process identity changed; no signal sent')
    controller.suspend()
    for child in processes[:-1]:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    gone,alive=psutil.wait_procs(processes[:-1],timeout=3)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    controller.kill()
    psutil.wait_procs([controller,*alive],timeout=3)
    time.sleep(.1)
    report.update(status='externally_drained_incomplete',
                  summary_sha256_after_stop=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  processes_still_alive=[p.pid for p in processes if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE])
    temporary=record.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(record)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
