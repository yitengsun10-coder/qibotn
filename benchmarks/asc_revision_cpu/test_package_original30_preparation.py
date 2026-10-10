import hashlib
import json
import zipfile

import pytest
from benchmarks.asc_revision_cpu import package_original30_preparation as packaging


def make_prior(path, *, mismatch=False, extra=False, name='STAGE_RECEIPT.json'):
    raw=json.dumps({'original30_complete':False,'formal_chain_supplement_complete':True}).encode()
    manifest={name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
    if mismatch:manifest[name]['sha256']='0'*64
    with zipfile.ZipFile(path,'x') as archive:
        archive.writestr(name,raw)
        archive.writestr('MANIFEST_SHA256.json',json.dumps(manifest))
        if extra:archive.writestr('undeclared.txt','no')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prior_validates_original_bytes_and_preserves_stage_receipt(tmp_path):
    path=tmp_path/'prior.zip';digest=make_prior(path)
    files,receipt=packaging.read_prior(path,digest)
    assert receipt['original30_complete'] is False
    assert 'prior_delivery/STAGE_RECEIPT.json' in files
    assert 'STAGE_RECEIPT.json' not in files


@pytest.mark.parametrize('kind',['hash','member','membership','path'])
def test_prior_rejects_changed_or_unsafe_data(tmp_path,kind):
    path=tmp_path/'prior.zip'
    digest=make_prior(path,mismatch=kind=='member',extra=kind=='membership',
                      name='../escape' if kind=='path' else 'STAGE_RECEIPT.json')
    if kind=='hash':digest='0'*64
    with pytest.raises(ValueError):packaging.read_prior(path,digest)


@pytest.mark.parametrize('name',['../x','/abs','C:/x','a\\b','./a','a//b','.'])
def test_names_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):packaging.safe_name(name)


@pytest.mark.parametrize('tests,failures,errors,skipped',[(0,0,0,0),(-1,0,0,0),(2,1,0,0),(2,0,1,0),(2,0,0,1)])
def test_failed_incomplete_or_empty_test_receipts_rejected(tmp_path,tests,failures,errors,skipped):
    path=tmp_path/'tests.xml'
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>')
    with pytest.raises(ValueError):packaging.test_totals(path)


def test_forecast_must_recompute_from_current_code(tmp_path,monkeypatch):
    from benchmarks.asc_revision_cpu import forecast_original30_shapes as model
    monkeypatch.setattr(model,'original30_plan',lambda:{'simulation_performed':False})
    path=tmp_path/'forecast.json';path.write_text('{"simulation_performed":true}')
    with pytest.raises(ValueError):packaging.validate_forecast(path)


def test_existing_delivery_never_overwritten(tmp_path):
    path=tmp_path/'keep.zip';path.write_bytes(b'existing user data')
    with pytest.raises(FileExistsError):packaging.package(tmp_path,'keep')
    assert path.read_bytes()==b'existing user data'
