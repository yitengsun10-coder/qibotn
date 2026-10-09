from copy import deepcopy
from benchmarks.asc_revision_cpu.drain_stalled_supplement import eligible


def test_only_completed_record_of_first_timeout_allows_drain():
    case={'kind':'chain','pass':False,'stop_reason':'wall_time_budget','wall_s':900.1}
    assert eligible({'cases':[case]})
    for name,value in [('kind','ring'),('pass',True),('stop_reason',None),('wall_s',899.9)]:
        changed=deepcopy(case);changed[name]=value
        assert not eligible({'cases':[changed]})
    assert not eligible({'cases':[]})
