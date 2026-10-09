"""Synthetic package gates; fixture ratios are not measured performance."""
import json
import pytest
from benchmarks.asc_revision_cpu.test_audit_resumed_common_policy import resumed_fixture
from benchmarks.asc_revision_cpu.test_audit_common_policy import common_fixture
from benchmarks.asc_revision_cpu.test_audit_supplement30 import suite_fixture
from benchmarks.asc_revision_cpu.audit_resumed_common_policy_chain import audit
from benchmarks.asc_revision_cpu.package_server724_stage import HERE,ROOT,sha,validate_chain_audit,package,repository_input


@pytest.fixture
def audit_receipt(resumed_fixture,tmp_path):
    data=audit(resumed_fixture)
    data['auditor_sha256']=sha((HERE/'audit_resumed_common_policy_chain.py').read_bytes())
    path=tmp_path/'chain_audit.json'
    path.write_text(json.dumps(data),encoding='utf-8')
    return path


def test_reaudit_matches_raw_evidence(resumed_fixture,audit_receipt):
    result=validate_chain_audit(resumed_fixture,audit_receipt)
    assert result['formal_chain_supplement_complete']
    assert not result['original30_complete'] and not result['supplement30_suite_complete']


@pytest.mark.parametrize('fault',['ratio','scope','auditor','raw'])
def test_reject_changed_delivery_evidence(resumed_fixture,audit_receipt,fault):
    data=json.loads(audit_receipt.read_text())
    if fault=='ratio':data['chain']['speedups_vs_official_common_qr_policy']['controlled_rank2']=7.
    elif fault=='scope':data['original30_complete']=True
    elif fault=='auditor':data['auditor_sha256']='0'*64
    elif fault=='raw':(resumed_fixture/'prior_segment/chain/run.log').write_bytes(b'changed')
    audit_receipt.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError):validate_chain_audit(resumed_fixture,audit_receipt)


def test_require_both_optional_inputs(tmp_path):
    with pytest.raises(ValueError):package(tmp_path,'fixture',tmp_path,None)


def test_relative_repository_input():
    assert repository_input('results')==ROOT/'results'


def test_reject_outside_repository_input():
    with pytest.raises(ValueError):repository_input(ROOT.parent/'outside-fixture')
