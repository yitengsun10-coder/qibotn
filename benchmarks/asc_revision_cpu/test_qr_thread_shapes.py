import pytest
from benchmarks.asc_revision_cpu.probe_large_qr_threads import parse_shapes


def test_declared_dimensions():
    assert parse_shapes(['1024x256','2048X1024'])==[(1024,256),(2048,1024)]


@pytest.mark.parametrize('values',[[],['2x3'],['2x0'],['-2x1'],['2x1','2x1'],['bad']])
def test_reject_ambiguous_or_unsafe_dimensions(values):
    with pytest.raises(ValueError):parse_shapes(values)
