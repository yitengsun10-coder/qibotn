import numpy as np
import pytest
from benchmarks.asc_revision_cpu.candidate_cached_qr import cached_qr, _workspace_sizes


@pytest.mark.parametrize('shape', [(1,1), (9,4), (4,9), (128,64), (256,128), (128,128)])
@pytest.mark.parametrize('complex_input', [False, True])
@pytest.mark.parametrize('singular', [False, True])
def test_exact_qr(shape, complex_input, singular):
    rng = np.random.default_rng(7361)
    matrix = rng.normal(size=shape)
    if complex_input:
        matrix = matrix + 1j * rng.normal(size=shape)
    if singular and shape[1] > 1:
        matrix[:, -1] = matrix[:, 0]
    before = matrix.copy()
    q, r = cached_qr(matrix)
    k = min(shape)
    assert q.shape == (shape[0], k) and r.shape == (k, shape[1])
    np.testing.assert_allclose(q @ r, before, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(k), rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(matrix, before)
    assert q.dtype == r.dtype == matrix.dtype
    assert not np.may_share_memory(q, matrix) and not np.may_share_memory(r, matrix)


def test_query_cache_has_only_scalar_sizes():
    _workspace_sizes.cache_clear()
    matrix = np.zeros((128, 64), dtype=np.complex128)
    cached_qr(matrix)
    cached_qr(matrix)
    assert _workspace_sizes.cache_info().hits >= 1
    assert all(isinstance(value, int) and value > 0 for value in _workspace_sizes(matrix.dtype.str, 128, 64))


def test_noncontiguous_views_remain_unchanged():
    rng = np.random.default_rng(193)
    backing = rng.normal(size=(256,128)) + 1j*rng.normal(size=(256,128))
    before = backing.copy()
    matrix = backing[::2,::2]
    q, r = cached_qr(matrix)
    np.testing.assert_allclose(q@r, matrix, atol=1e-12)
    np.testing.assert_array_equal(backing, before)
