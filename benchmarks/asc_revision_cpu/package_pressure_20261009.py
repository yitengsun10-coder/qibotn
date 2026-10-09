"""Verify and deliver exact CPU pressure evidence without an all-done claim."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = Path('C:/Users/孙逸腾/Desktop/超算')
STEM = 'QiboTN_精确20比特实测与30比特待验收_20261009'
CORE = ('quimb.py', '_exact_numpy.py', '_linear_cnot.py', '_structured_numpy.py',
        '_cnot_schedule.py', '_mixed_schedule.py')
TESTS = ('linear_cnot', 'adaptive_cnot', 'exact_numpy', 'shared_expectation', 'quimb_backend',
         'isometric_numpy', 'lookahead_routing', 'structured_numpy', 'cnot_schedule', 'owned_buffers', 'mixed_schedule')
HELPERS = ('README.md', 'PRESSURE_RESULTS_20261009.md', 'INTEGRATED_UPDATE_20261009.md',
    'PRESSURE_PROGRESS_20261009.md', 'EXACT30_PREPARATION_20261009.md',
    'PROFILE_AND_ACCEPTANCE_PLAN_20261009.md', 'environment_20261009.json',
    'paired_pressure_chain.py', 'resume_pressure_chain.py', 'extend_pressure_supervisor.py',
    'audit_pressure_chain.py', 'test_pressure_chain.py', 'test_pressure_resume.py',
    'candidate_cached_qr.py', 'test_cached_qr.py', 'exact30_acceptance.py', 'test_exact30_acceptance.py',
    'capacity_plan.py', 'upstream_reference.py', 'evaluate.py', 'probe_linear_holdout.py',
    'paired_integrated_mixed.py', 'paired_upstream_mixed.py', 'paired_phase_insertions.py',
    'candidate_mixed_schedule.py', 'candidate_phase_blocks.py', 'candidate_phase_insertions.py',
    'test_mixed_schedule.py', 'test_phase_blocks.py', 'test_phase_insertions.py',
    'audit_upstream_mixed.py', 'test_upstream_mixed_audit.py', 'probe_large_bounded.py',
    'integrated_mixed_audit_20261009.json', 'phase_holdout_snapshot_audit_20261009.json',
    'phase_insertion_snapshot_audit_20261009.json', 'package_pressure_20261009.py',
    'validation_tests_integrated_mixed_20261009.xml', 'validation_tests_research_and_audit_20261009.xml',
    'validation_pressure_resume_cached_qr_20261009.xml')
DATASETS = ('integrated_mixed16', 'phase_holdout16', 'phase_insertion_holdout16',
            'large30_resource_probe', 'pressure20_calibration', 'pressure20_9991',
            'pressure20_9991_resumed', 'exact30_preparation_plan')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_junit(path, expected):
    suite = ET.parse(path).getroot().find('testsuite')
    counts = {key: int(suite.attrib[key]) for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != {'tests': expected, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError(f'Unexpected JUnit: {path}: {counts}')
    return counts


def main():
    sys.path.insert(0, str(ROOT))
    from benchmarks.asc_revision_cpu.audit_pressure_chain import audit as audit_pressure
    from benchmarks.asc_revision_cpu.audit_upstream_mixed import audit as audit_mixed
    pressure = audit_pressure(ROOT / 'results/revision_cpu_20261009_pressure20_9991_resumed')
    mixed = audit_mixed(ROOT / 'results/revision_cpu_20261009_integrated_mixed16')
    if pressure['formal_30q_acceptance_complete'] or mixed['all_requirements_complete']:
        raise ValueError('Do not label stage evidence as all-done')
    junit_paths = ((HERE / 'validation_tests_integrated_mixed_20261009.xml', 322),
        (HERE / 'validation_tests_research_and_audit_20261009.xml', 73),
        (HERE / 'validation_pressure_resume_cached_qr_20261009.xml', 50),
        (ROOT / 'results/validation_exact30_preparation_20261009.xml', 18))
    receipt = {'status': 'verified_stage_only', 'pressure20': pressure, 'mixed16': mixed,
        'tests': {str(path.relative_to(ROOT)): checked_junit(path, expected) for path, expected in junit_paths},
        'formal_30q_acceptance_complete': False, 'seven_x_achieved': False,
        'word_pdf_synchronized': False, 'all_contest_tasks_complete': False,
        'github_publication_status': 'separate publication receipt required', 'submission_sent': False}
    receipt_path = HERE / 'pressure_verified_stage_receipt_20261009.json'
    if receipt_path.exists():
        raise FileExistsError('Preserve prior receipt')
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = set()
    def add(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    for name in CORE:
        add(ROOT / 'src/qibotn/backends' / name)
    for name in TESTS:
        add(ROOT / 'tests' / ('test_' + name + '.py'))
    for name in HELPERS:
        add(HERE / name)
    add(receipt_path)
    add(ROOT / 'benchmarks/asc_large_cpu/benchmark.py')
    for name in ('RESULTS.md', 'final_report/WMA_QiboTN_final_report_20261006.md',
                 'final_report/requirements_audit_20261006.md'):
        add(ROOT / 'benchmarks/asc_large_cpu' / name)
    for name in ('pressure20_final_audit_20261009.json', 'validation_exact30_preparation_20261009.xml'):
        add(ROOT / 'results' / name)
    for dataset in DATASETS:
        folder = ROOT / ('results/revision_cpu_20261009_' + dataset)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                add(path)
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(paths)}
    pdf = Path('C:/Users/孙逸腾/Documents/WeChat Files/wxid_mcrolp1g1yp622/FileStorage/File/2026-09/Qibotn.pdf')
    payload = pdf.read_bytes()
    if sha(payload) != '630ee41393f8c1312d76260aa705d7c6bda17dbed4a0cf242bd282874f409af9':
        raise ValueError('Requirement PDF identity mismatch')
    files['requirements/Qibotn.pdf'] = payload
    manifest = {'status': 'stage_not_final_acceptance', 'formal_30q_complete': False,
        'members': {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    DEST.mkdir(parents=True, exist_ok=True)
    archive_path, report_path = DEST / (STEM + '.zip'), DEST / (STEM + '.md')
    hash_path, delivery_path = DEST / (STEM + '.zip.sha256'), DEST / (STEM + '_交付核验.json')
    if any(path.exists() for path in (archive_path, report_path, hash_path, delivery_path)):
        raise FileExistsError('Refusing to overwrite old delivery')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr('MANIFEST_SHA256.json', manifest_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or set(names) != {*files, 'MANIFEST_SHA256.json'} or
                archive.testzip() is not None or archive.read('MANIFEST_SHA256.json') != manifest_bytes):
            raise ValueError('ZIP membership/manifest/CRC mismatch')
        for name, data in files.items():
            if sha(archive.read(name)) != sha(data):
                raise ValueError('ZIP member hash mismatch: ' + name)
    archive_hash = sha(archive_path.read_bytes())
    hash_path.write_text(archive_hash + '  ' + archive_path.name + '\n', encoding='utf-8')
    shutil.copyfile(HERE / 'PRESSURE_RESULTS_20261009.md', report_path)
    if sha(report_path.read_bytes()) != sha((HERE / 'PRESSURE_RESULTS_20261009.md').read_bytes()):
        raise ValueError('Desktop report byte mismatch')
    delivery = {'status': 'verified_stage_delivery', 'archive': str(archive_path), 'sha256': archive_hash,
        'report': str(report_path), 'member_hashes_verified': len(files), 'members_including_manifest': len(files)+1,
        'crc_verified': True, 'exact30_complete': False, 'all_requirements_complete': False,
        'note': 'Publication, Word/PDF and final submission are separate outstanding steps.'}
    delivery_path.write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(delivery, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
