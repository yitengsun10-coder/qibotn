"""Append verified local preparation to the immutable accepted stage package.

No remote completion is inferred, no running directory is archived, and no
diagnostic or shape forecast is promoted to original30 acceptance or speed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIOR_SHA = '7826ec23b914a79dee00dd4258de652d7ae792aa5974efa603fa0491c408fd38'
PRIOR = Path('C:/Users/孙逸腾/Desktop/超算/QiboTN_724补充链式30比特实测与原线路待验收_20261009_2315.zip')
REPORT = HERE/'ORIGINAL30_LOCAL_PREPARATION_20261010.md'
TEST_RECEIPT = ROOT/'results/validation_original30_local_preparation_20261010.xml'
PACKAGE_TEST_RECEIPT = ROOT/'results/validation_original30_package_guards_20261010.xml'
FORECAST = ROOT/'results/original30_shape_forecast_v3_20261010.json'
HELPERS = (
    'candidate_short_cnot_rank2.py', 'test_short_cnot_rank2.py',
    'probe_short_cnot_rank2.py', 'launch_original30_probe.py',
    'common_mps_thread_policy.py', 'test_common_mps_policy.py',
    'probe_short_mps_policy.py', 'audit_original30_diagnostics.py',
    'test_audit_original30_diagnostics.py', 'forecast_original30_shapes.py',
    'test_forecast_original30_shapes.py', 'package_original30_preparation.py',
    'test_package_original30_preparation.py', 'ORIGINAL30_LOCAL_PREPARATION_20261010.md',
    'SERVER724_RANK2_STAGE_20261009.md')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member')
    if path.as_posix() != name or name.endswith('/'):
        raise ValueError('Noncanonical archive member')
    return name


def read_prior(path, expected_sha=PRIOR_SHA):
    if sha(path.read_bytes()) != expected_sha:
        raise ValueError('Prior package SHA256 differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('Prior ZIP duplicate/CRC failure')
        for name in names:
            safe_name(name)
        manifest = json.loads(archive.read('MANIFEST_SHA256.json'))
        if set(names) != set(manifest)|{'MANIFEST_SHA256.json'}:
            raise ValueError('Prior ZIP membership differs')
        files = {}
        for name, metadata in manifest.items():
            raw = archive.read(name)
            if len(raw) != metadata['bytes'] or sha(raw) != metadata['sha256']:
                raise ValueError('Prior member hash/size differs')
            files[name] = raw
    prior_receipt = json.loads(files['STAGE_RECEIPT.json'])
    if prior_receipt['original30_complete'] is not False:
        raise ValueError('Prior stage must not claim original30 completion')
    files['prior_delivery/STAGE_RECEIPT.json'] = files.pop('STAGE_RECEIPT.json')
    return files, prior_receipt


def test_totals(path):
    suites = ET.parse(path).getroot().findall('testsuite')
    totals = {key:sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if totals['tests'] <= 0 or any(totals[key] for key in ('failures','errors','skipped')):
        raise ValueError('Latest local tests are not fully passed')
    return totals


def validate_forecast(path):
    from benchmarks.asc_revision_cpu.forecast_original30_shapes import original30_plan
    report = json.loads(path.read_text(encoding='utf-8'))
    if report != original30_plan():
        raise ValueError('Shape forecast no longer matches current implementation')
    return report


def package(destination, stem, prior=PRIOR):
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination/(stem+'.zip')
    report_path = destination/(stem+'.md')
    receipt_path = destination/(stem+'_交付核验.json')
    sha_path = destination/(stem+'.zip.sha256')
    targets = (archive_path, report_path, receipt_path, sha_path)
    if any(path.exists() for path in targets):
        raise FileExistsError('Refusing to replace an old delivery')
    files, prior_receipt = read_prior(prior)
    tests = test_totals(TEST_RECEIPT)
    if tests['tests'] != 153:
        raise ValueError('Unexpected local regression test count')
    package_tests = test_totals(PACKAGE_TEST_RECEIPT)
    forecast = validate_forecast(FORECAST)

    def add(path):
        resolved = path.resolve()
        name = safe_name(resolved.relative_to(ROOT).as_posix())
        files[name] = resolved.read_bytes()

    for name in HELPERS:
        add(HERE/name)
    add(ROOT/'.gitattributes')
    add(TEST_RECEIPT)
    add(PACKAGE_TEST_RECEIPT)
    add(FORECAST)
    # Preserve first failed fixture receipts and intermediate successful runs.
    for pattern in ('validation_short*20261009.xml',
                    'validation_audit_original30*20261010.xml',
                    'validation_forecast_original30*20261010.xml',
                    'original30_shape_forecast*20261010.json'):
        for path in (ROOT/'results').glob(pattern):
            add(path)
    for folder in ('original30_pilot_20261009_2332',
                   'short_cnot_probe_smoke_local_20261009',
                   'plain_common_probe_smoke_local_20261009',
                   'short_mps_policy_smoke_local_20261009'):
        root = ROOT/'results'/folder
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob('*'):
            if path.is_file():
                add(path)
    # Reject silently packaging a different production version than the one
    # delivered before. Keep the unrelated dirty pyproject working file out.
    for name in ('quimb.py','_exact_numpy.py','_linear_cnot.py',
                 '_structured_numpy.py','_cnot_schedule.py','_mixed_schedule.py'):
        path = ROOT/'src/qibotn/backends'/name
        if files[path.relative_to(ROOT).as_posix()] != path.read_bytes():
            raise ValueError('Production source changed since prior validated stage')
    committed_config = subprocess.check_output(['git','show','HEAD:pyproject.toml'], cwd=ROOT)
    if files['pyproject.toml'] != committed_config:
        raise ValueError('Committed project configuration changed')
    receipt = {
        'status':'verified_local_preparation_not_final_acceptance',
        'prior_package_sha256':PRIOR_SHA,
        'prior_formal_chain_supplement_complete':prior_receipt['formal_chain_supplement_complete'],
        'local_regression_tests':tests,
        'package_guard_tests':package_tests,
        'shape_forecast_sha256':sha(FORECAST.read_bytes()),
        'shape_forecast_is_not_observed_memory':True,
        'original30_complete':False,
        'original30_publishable_speedup':None,
        'supplement30_suite_complete':False,
        'new_research_production_integrated':False,
        'word_pdf_updated':False,
        'submission_sent':False,
        'unretrieved_remote_pilot_raw_evidence_included':False,
        'last_remote_mps_pilot_final_status_verified':False,
        'unrelated_pyproject_edits_preserved':True,
        'forecast_scope':forecast['status'],
    }
    files['STAGE_RECEIPT.json'] = json.dumps(receipt,ensure_ascii=False,indent=2).encode('utf-8')
    manifest = {name:{'bytes':len(raw),'sha256':sha(raw)} for name,raw in sorted(files.items())}
    manifest_raw = json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):
            archive.writestr(name,raw)
        archive.writestr('MANIFEST_SHA256.json',manifest_raw)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files)|{'MANIFEST_SHA256.json'}:
            raise ValueError('New ZIP membership/CRC failure')
        if len(archive.namelist()) != len(files)+1:
            raise ValueError('New ZIP has duplicate members')
        if archive.read('MANIFEST_SHA256.json') != manifest_raw:
            raise ValueError('New ZIP manifest changed')
        for name,raw in files.items():
            if archive.read(name) != raw:
                raise ValueError('New ZIP member bytes differ')
    with report_path.open('xb') as stream:
        stream.write(REPORT.read_bytes())
    receipt.update(zip_sha256=sha(archive_path.read_bytes()),crc_verified=True,
        member_hashes_verified=len(files),members_including_manifest=len(files)+1,
        archive=str(archive_path),report=str(report_path))
    with receipt_path.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    with sha_path.open('x',encoding='utf-8') as stream:
        stream.write(receipt['zip_sha256']+'  '+archive_path.name+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('C:/Users/孙逸腾/Desktop/超算'))
    parser.add_argument('--stem',default='QiboTN_原30比特优化与资源预检阶段包_20261010')
    args = parser.parse_args()
    print(json.dumps(package(args.destination,args.stem),ensure_ascii=False,indent=2))
