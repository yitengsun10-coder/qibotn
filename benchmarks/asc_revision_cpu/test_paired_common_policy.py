from copy import deepcopy
import math
import pytest
from benchmarks.asc_revision_cpu.paired_supplement30_common_policy import METHODS,summarize_rows


def rows():
    return [{'method':m,'repeat':r,'runtime_s':10. if m==METHODS[0] else 2.,'abs_error':1e-14}
            for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]


def test_complete_rows_and_correct_scope():
    result=summarize_rows(rows())
    assert result['speedups_vs_official_common_qr_policy']['controlled_rank2']==5.
    assert result['rank2_speedup_vs_integrated_common_qr_policy']==1.


@pytest.mark.parametrize('fault',['missing','extra','reorder','error','nan','zero'])
def test_reject_partial_or_bad_rows(fault):
    data=deepcopy(rows())
    if fault=='missing':data.pop()
    elif fault=='extra':data.append(data[0])
    elif fault=='reorder':data.reverse()
    elif fault=='error':data[0]['abs_error']=.01
    elif fault=='nan':data[0]['runtime_s']=math.nan
    elif fault=='zero':data[0]['runtime_s']=0
    with pytest.raises(ValueError):summarize_rows(data)
