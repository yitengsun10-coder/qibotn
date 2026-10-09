"""Research-only LAPACK QR workspace-query caching, with owned inputs.

The same Householder QR/economy Q is computed as SciPy, not a Cholesky or
approximate factorization. Only scalar workspace sizes and function handles
are cached. No tensor values, factors or outputs survive between calls.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg.lapack import get_lapack_funcs
from qibotn.backends._exact_numpy import _reduced_qr as prior_qr


@lru_cache(maxsize=4)
def _functions(dtype_string):
    sample = np.empty((1, 1), dtype=np.dtype(dtype_string), order='F')
    return get_lapack_funcs(('geqrf', 'orgqr'), (sample,))


@lru_cache(maxsize=64)
def _workspace_sizes(dtype_string, rows, columns):
    geqrf, orgqr = _functions(dtype_string)
    workspace = np.empty((rows, columns), dtype=np.dtype(dtype_string), order='F')
    _, tau, work, info = geqrf(workspace, lwork=-1, overwrite_a=1)
    if info != 0:
        raise ValueError('LAPACK geqrf workspace query failed')
    factor_work = int(work[0].real)
    _, work, info = orgqr(workspace, tau, lwork=-1, overwrite_a=1)
    if info != 0:
        raise ValueError('LAPACK orgqr workspace query failed')
    return factor_work, int(work[0].real)


def cached_qr(matrix):
    rows, columns = matrix.shape
    if rows < columns or min(rows, columns) < 64 or matrix.dtype not in (np.dtype('float64'), np.dtype('complex128')):
        return prior_qr(matrix)
    # Matrix may be a mutable view of a live MPS. LAPACK may overwrite only
    # this private owned Fortran copy, never an input or published factor.
    workspace = np.array(matrix, order='F', copy=True)
    geqrf, orgqr = _functions(workspace.dtype.str)
    factor_work, q_work = _workspace_sizes(workspace.dtype.str, rows, columns)
    factor, tau, _, info = geqrf(workspace, lwork=factor_work, overwrite_a=1)
    if info != 0:
        raise ValueError('LAPACK geqrf failed')
    r = np.triu(factor[:columns, :])
    q, _, info = orgqr(factor, tau, lwork=q_work, overwrite_a=1)
    if info != 0:
        raise ValueError('LAPACK orgqr failed')
    return q, r
