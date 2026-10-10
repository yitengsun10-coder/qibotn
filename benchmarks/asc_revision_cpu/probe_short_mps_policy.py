"""Original30 bounded diagnosis: common gate-size BLAS + common QR policy.

Both plain and short-order research methods receive the identical policy.
This is not formal paired timing, complete-state acceptance or a speed claim.
The unchanged previous worker/supervisor still own fixed input and safeguards.
"""
import argparse
import math
from pathlib import Path

from benchmarks.asc_revision_cpu import probe_short_cnot_rank2 as previous
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import clone
from benchmarks.asc_revision_cpu.common_mps_thread_policy import common_mps_policy
from benchmarks.asc_revision_cpu.profile_fixed30_bounded import save

SOURCES = [*previous.SOURCES, Path(__file__),
           Path(__file__).with_name('common_mps_thread_policy.py')]
suite = clone(previous.suite, {'SOURCES': SOURCES, '__file__': __file__})


def worker(args):
    with common_mps_policy() as calls:
        def record(path, value):
            if path.name == 'profile.json':
                value = {**value, 'thread_policy':
                    'common CircuitMPS gate bond-width BLAS1/4/8/16; common inner QR width1/4/8/16',
                    'common_mps_gate_calls': dict(calls),
                    'thread_policy_uses_outputs_or_timings': False}
            save(path, value)
        nested_worker = clone(previous.worker, {'save': record})
        return nested_worker(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['original30'], default='original30')
    parser.add_argument('--candidate', choices=['plain', 'short'], required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--memory-mib', type=float, default=32768)
    parser.add_argument('--reserve-mib', type=float, default=2048)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.memory_mib, args.reserve_mib)):
        parser.error('Finite positive resource budgets required')
    if args.threads != 16 and not (args.smoke and 1 <= args.threads <= 16):
        parser.error('Original30 diagnosis requires common 16-CPU affinity')
    raise SystemExit(worker(args) if args.worker else suite(args))
