"""Detach exactly one bounded original30 research probe and record its identity.

No automatic reruns, competing benchmarks, purchases or shutdowns. The child
probe owns wall-time/process-tree/cgroup protection and its frozen evidence.
"""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import psutil
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import ROOT, effective_memory, save


def launch(candidate, output, seconds=180., memory_mib=32768., reserve_mib=2048., policy='qr'):
    if candidate not in {'short', 'plain'} or not 0 < seconds <= 300:
        raise ValueError('Only declared candidates and at most 300 seconds per probe')
    if memory_mib != 32768 or reserve_mib != 2048:
        raise ValueError('Fixed 32-GiB process-tree / 2-GiB reserve policy')
    if policy not in {'qr', 'mps'}:
        raise ValueError('Only declared common QR / common MPS policies')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must be inside this repository results directory')
    metadata = output.with_suffix('.launcher.json')
    log = output.with_suffix('.controller.log')
    if any(path.exists() for path in (output, metadata, log)):
        raise ValueError('Fresh output, launcher receipt and log required')
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid() or 'python' not in (process.info['name'] or '').lower():
            continue
        tokens = process.info['cmdline'] or []
        if any('asc_revision_cpu/' in token.replace('\\', '/') and
               token.endswith('.py') and 'launch_original30_probe' not in token for token in tokens):
            raise ValueError('Another revision benchmark/helper is active; no parallel launch')
    memory = effective_memory(psutil.virtual_memory().available)
    if memory['available_bytes'] < (memory_mib + reserve_mib) * 2**20:
        raise ValueError('Cgroup-aware memory admission failed')
    entry = ROOT / 'benchmarks/asc_revision_cpu' / (
        'probe_short_cnot_rank2.py' if policy=='qr' else 'probe_short_mps_policy.py')
    command = [sys.executable, '-X', 'utf8', str(entry), '--candidate', candidate,
        '--output', str(output), '--seconds', str(seconds), '--threads', '16',
        '--memory-mib', str(memory_mib), '--reserve-mib', str(reserve_mib), '--execute']
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'recorded_at': datetime.now().astimezone().isoformat(), 'pid': child.pid,
        'process_creation_time': psutil.Process(child.pid).create_time(), 'command': command,
        'candidate': candidate, 'policy': policy, 'output': str(output), 'log': str(log), 'memory_at_launch': memory,
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'probe_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
        'formal_30q_acceptance_complete': False, 'publishable_speedup': None}
    save(metadata, record)
    return record


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['short', 'plain'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--policy', choices=['qr', 'mps'], default='qr')
    args = parser.parse_args()
    print(json.dumps(launch(args.candidate, args.output, args.seconds, policy=args.policy), indent=2))
