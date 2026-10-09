"""Identical matrix-size-only QR thread policy for reference and candidates.

Only thread scheduling is wrapped: native Quimb and current NumPy/SciPy QR
algorithms, arithmetic and input matrices are unchanged. Caller BLAS limits
and registered functions are restored even on failure. Single-worker only.
"""
from contextlib import contextmanager
from functools import wraps
from threadpoolctl import ThreadpoolController


def policy(rows,columns):
    width=min(rows,columns)
    if width<512:return 1
    if width<1024:return 4
    if width<2048:return 8
    return 16


@contextmanager
def common_qr_policy():
    from quimb.tensor import decomp
    from qibotn.backends import _exact_numpy as exact
    controller=ThreadpoolController()
    native=decomp._SPLIT_FNS['qr']
    kernel=exact._reduced_qr
    calls={1:0,4:0,8:0,16:0}
    def wrap(original):
        @wraps(original)
        def scheduled(matrix,*args,**kwargs):
            threads=policy(*matrix.shape[-2:]);calls[threads]+=1
            with controller.limit(limits=threads,user_api='blas'):
                return original(matrix,*args,**kwargs)
        return scheduled
    native_wrapper,kernel_wrapper=wrap(native),wrap(kernel)
    decomp._SPLIT_FNS['qr']=native_wrapper
    exact._reduced_qr=kernel_wrapper
    try:
        yield calls
    finally:
        decomp._SPLIT_FNS['qr']=native
        exact._reduced_qr=kernel
