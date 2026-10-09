from copy import deepcopy
import math
import pytest
from benchmarks.asc_revision_cpu.paired_supplement30 import METHODS, check_state, summarize


def rows():
    return [{'method':m,'repeat':r,'runtime_s':10. if m==METHODS[0] else 2.,'abs_error':1e-14}
            for r in range(4) for m in (METHODS if r%2==0 else METHODS[::-1])]


def test_complete_rows():
    result=summarize(rows())
    assert result['speedups_vs_tuned_official']['integrated']==5.
    assert result['candidate_speedup_vs_integrated']==1.


@pytest.mark.parametrize('kind',['missing','extra','reorder','error','nan','zero'])
def test_reject_partial_or_wrong_rows(kind):
    data=deepcopy(rows())
    if kind=='missing': data.pop()
    elif kind=='extra': data.append(data[0])
    elif kind=='reorder': data[0],data[1]=data[1],data[0]
    elif kind=='error': data[2]['abs_error']=1e-3
    elif kind=='nan': data[2]['runtime_s']=math.nan
    elif kind=='zero': data[2]['runtime_s']=0.
    with pytest.raises(ValueError): summarize(data)


def test_phase_equivalent_state():
    assert check_state(1.,1.,1j)['full_state_comparison_pass']


@pytest.mark.parametrize('actual,overlap',[(0.,0.),(.9,.9),(1.,.9),(1.,math.nan)])
def test_reject_bad_complete_state(actual,overlap):
    with pytest.raises(ValueError): check_state(1.,actual,complex(overlap))
