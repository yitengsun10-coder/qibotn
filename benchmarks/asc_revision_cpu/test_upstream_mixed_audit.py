from copy import deepcopy
import pytest
from benchmarks.asc_revision_cpu.audit_upstream_mixed import METHODS, validate_result


def completed():
    return {"status": "pass", "precision": "complex128", "cutoff": 0, "max_bond": None,
            "measured_repeats": 3, "warmup_repeats": 1, "threads": 1,
            "reference": {"extra_norm_contraction": False, "state_preparations_per_call": 1},
            "rows": [{"method": m, "repeat": k, "runtime_s": 1., "value": .2,
                      "exact_value": .2, "abs_error": 0.} for m in METHODS for k in range(4)],
            "means_s": {m: 1. for m in METHODS},
            "full_state": {m: {"norm_squared": 1., "fidelity": 1., "max_state_error": 0.}
                           for m in METHODS[1:]}}, {"status": "pass", "exit_code": 0, "stop_reason": None}


def test_complete():
    data, summary = completed()
    assert validate_result(data, summary)["rows_verified"] == 16


def test_integrated_three_path_comparison():
    data, summary = completed()
    methods = ("upstream_typo_only", "prior_default", "integrated")
    data["rows"] = [{"method": m, "repeat": k, "runtime_s": 1., "value": .2,
                     "exact_value": .2, "abs_error": 0.} for m in methods for k in range(4)]
    data["means_s"] = {m: 1. for m in methods}
    data["full_state"] = {m: {"norm_squared": 1., "fidelity": 1., "max_state_error": 0.}
                          for m in methods[1:]}
    result = validate_result(data, summary, methods)
    assert result["rows_verified"] == 12
    assert result["full_state_methods_verified"] == 2


@pytest.mark.parametrize("failure", ["partial", "duplicate", "norm_reference", "state", "mean", "exit", "truncation"])
def test_reject_invalid_evidence(failure):
    data, summary = completed()
    data = deepcopy(data)
    if failure == "partial":
        data["rows"].pop()
    elif failure == "duplicate":
        data["rows"][-1] = data["rows"][0]
    elif failure == "norm_reference":
        data["reference"]["extra_norm_contraction"] = True
    elif failure == "state":
        data["full_state"]["candidate"]["fidelity"] = .28
    elif failure == "mean":
        data["means_s"]["candidate"] = .001
    elif failure == "exit":
        summary["exit_code"] = 1
    else:
        data["cutoff"] = 1e-12
    with pytest.raises(ValueError):
        validate_result(data, summary)
