"""Common metadata-only gate BLAS policy for all Quimb CircuitMPS methods.

One-site gates stay single threaded. Two/multi-site gates use the same bounded
1/4/8/16 rule based on the current MPS bond width. An inner common_qr_policy
can still choose a smaller QR team. No gate values, output or timing decisions.
Single-worker diagnostic policy; restore the registered method on all exits.
"""
from contextlib import contextmanager
from functools import wraps
from threadpoolctl import ThreadpoolController

from benchmarks.asc_revision_cpu.common_qr_thread_policy import policy


@contextmanager
def common_mps_policy():
    from quimb.tensor import CircuitMPS
    original = CircuitMPS._apply_gate
    owned = '_apply_gate' in vars(CircuitMPS)
    raw = vars(CircuitMPS).get('_apply_gate')
    controller = ThreadpoolController()
    calls = {1: 0, 4: 0, 8: 0, 16: 0}
    @wraps(original)
    def scheduled(self, gate, *args, **kwargs):
        threads = 1
        if len(gate.qubits or ()) > 1:
            width = self._psi.max_bond()
            threads = policy(width, width)
        calls[threads] += 1
        with controller.limit(limits=threads, user_api='blas'):
            return original(self, gate, *args, **kwargs)
    CircuitMPS._apply_gate = scheduled
    try:
        yield calls
    finally:
        if owned:
            CircuitMPS._apply_gate = raw
        else:
            delattr(CircuitMPS, '_apply_gate')
