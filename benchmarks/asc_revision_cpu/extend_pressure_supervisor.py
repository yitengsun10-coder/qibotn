"""Take over only the verified local pressure worker without interrupting it.

The human authorized a 1--2 hour continuation. The old supervisory Python
process is replaced, not the computation. Windows process handles preserve
the real worker exit code. The handoff and original supervisory exit are
explicit evidence, not a claim of an initially clean outer exit.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def identity(process):
    return {'pid': process.pid, 'created_at_epoch': process.create_time(),
            'command': process.cmdline()}


def matches(process_id, created):
    try:
        return abs(psutil.Process(process_id).create_time() - created) < .01
    except psutil.NoSuchProcess:
        return False


def main(args):
    if os.name != 'nt':
        raise RuntimeError('This handoff uses Windows process handles')
    directory = args.directory.resolve()
    metadata = json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    if result['calibration_only'] or result['status'] == 'pass':
        raise RuntimeError('Expected the one active non-calibration pressure run')
    if (directory / 'supervisor_handoff.json').exists() or (directory / 'extended_supervisor_summary.json').exists():
        raise RuntimeError('Do not duplicate or overwrite a previous handoff')
    if (result['qubits'], result['layers'], result['gates'], result['depth']) != (20,169,9991,677):
        raise RuntimeError('Unexpected computation scope')
    for relative, digest in result['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Computation sources changed; do not take over')
    supervisor = psutil.Process(metadata['controller_pid'])
    command = supervisor.cmdline()
    if '--worker' in command or not any('paired_pressure_chain.py' in arg for arg in command):
        raise RuntimeError('Controller PID does not belong to the expected harness')
    children = supervisor.children(recursive=True)
    workers = [p for p in children if '--worker' in p.cmdline() and
               any(directory.name in arg for arg in p.cmdline()) and
               any('paired_pressure_chain.py' in arg for arg in p.cmdline())]
    if not workers:
        raise RuntimeError('No live owned computation worker found')
    # A Windows virtualenv launcher has the same command but only a few MiB.
    # The actual interpreter is the owned descendant with the highest RSS.
    worker = max(workers, key=lambda p: p.memory_info().rss)
    if worker.memory_info().rss < 30 * 2**20:
        raise RuntimeError('Could not identify the actual computation interpreter')
    worker_identity = identity(worker)
    supervisor_identity = identity(supervisor)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, worker.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    took_over = finished = False
    report = None
    begin = time.monotonic()
    initial_elapsed = max(0., time.time() - supervisor.create_time())
    peak, reason = worker.memory_info().rss, None
    handoff = {'status': 'prepared', 'recorded_at': datetime.now().astimezone().isoformat(),
        'human_authorization': '保持电脑开机，继续完整验证',
        'original_supervisor': supervisor_identity, 'continuous_computation_worker': worker_identity,
        'new_supervisor': identity(psutil.Process()),
        'deadline_seconds_from_original_start': args.seconds,
        'memory_budget_mib_unchanged': args.memory_mib,
        'computation_restarted': False, 'old_timing_rows_reused_from_another_version': False,
        'handoff_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    shutil.copyfile(Path(__file__), directory / 'extend_pressure_supervisor_snapshot.py')
    save(directory / 'supervisor_handoff.json', handoff)
    try:
        # Windows TerminateProcess does not terminate descendants. Only the
        # controller above was selected; the CPU worker and launcher survive.
        supervisor.terminate()
        handoff['original_supervisor_exit_code'] = supervisor.wait(timeout=10)
        if not matches(worker.pid, worker_identity['created_at_epoch']):
            raise RuntimeError('Worker ended during handoff; inspect its final result')
        took_over = True
        handoff['status'] = 'active_supervisor_replaced_worker_continuous'
        save(directory / 'supervisor_handoff.json', handoff)
        print(json.dumps(handoff, ensure_ascii=False), flush=True)
        last_checkpoint = 0.
        while kernel.WaitForSingleObject(handle, 0) == 258:
            try:
                tree = psutil.Process(worker.pid)
                if abs(tree.create_time() - worker_identity['created_at_epoch']) > .01:
                    raise RuntimeError('Worker PID was reused')
                current_rss = tree.memory_info().rss + sum(c.memory_info().rss for c in tree.children(recursive=True))
                peak = max(peak, current_rss)
                elapsed = initial_elapsed + time.monotonic() - begin
                if current_rss > args.memory_mib * 2**20:
                    reason = 'process_tree_memory_budget_exceeded'
                elif psutil.virtual_memory().available < 1000 * 2**20:
                    reason = 'system_available_memory_below_1000_MiB'
                elif elapsed > args.seconds:
                    reason = 'user_authorized_extended_wall_clock_budget_exceeded'
                if reason:
                    for child in tree.children(recursive=True):
                        child.kill()
                    tree.kill()
                if time.monotonic() - last_checkpoint > 5:
                    save(directory / 'extended_supervisor_checkpoint.json', {
                        'status': 'running' if reason is None else 'stopping',
                        'elapsed_s_from_original_start': elapsed,
                        'peak_rss_mib_since_handoff': peak / 2**20,
                        'worker_pid': worker.pid, 'stop_reason': reason})
                    last_checkpoint = time.monotonic()
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        final = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        passed = code.value == 0 and reason is None and final.get('status') == 'pass'
        report = {'status': 'pass' if passed else 'incomplete_or_failed', 'exit_code': code.value,
            'exit_code_scope': 'real continuous computation interpreter, not the original supervisor',
            'stop_reason': reason, 'calibration_only': False, 'formal_30q_acceptance_complete': False,
            'elapsed_s_from_original_start': initial_elapsed + time.monotonic() - begin,
            'peak_rss_mib_since_handoff': peak / 2**20,
            'original_supervisor_exit_code': handoff['original_supervisor_exit_code'],
            'original_supervisor_replaced': True, 'computation_restarted': False,
            'budgets': {'seconds_from_original_start': args.seconds, 'memory_mib': args.memory_mib}}
        save(directory / 'extended_supervisor_summary.json', report)
        handoff['status'] = 'completed' if passed else 'completed_with_failure_or_stop'
        save(directory / 'supervisor_handoff.json', handoff)
        finished = True
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        # Never leave an owned expensive process without a live safety guard.
        if took_over and not finished and matches(worker.pid, worker_identity['created_at_epoch']):
            owned = psutil.Process(worker.pid)
            for child in owned.children(recursive=True):
                child.kill()
            owned.kill()
            save(directory / 'extended_supervisor_failure.json', {
                'status': 'supervisor_exception_owned_worker_stopped',
                'recorded_at': datetime.now().astimezone().isoformat(),
                'worker': worker_identity})
        kernel.CloseHandle(handle)
    if report is None or report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seconds', type=float, default=7200)
    parser.add_argument('--memory-mib', type=float, default=1800)
    args = parser.parse_args()
    main(args)
