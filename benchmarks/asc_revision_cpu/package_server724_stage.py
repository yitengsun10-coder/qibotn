"""Package checked stage bytes, never a final-acceptance or live-run archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def repository_input(path):
    resolved=(ROOT/path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def validate_chain_audit(root,audit_path):
    from benchmarks.asc_revision_cpu.audit_resumed_common_policy_chain import audit
    saved=json.loads(audit_path.read_text(encoding='utf-8'))
    fresh=audit(root)
    fresh['auditor_sha256']=sha((HERE/'audit_resumed_common_policy_chain.py').read_bytes())
    if saved!=fresh or fresh['status']!='pass_explicit_chain_continuation_only':
        raise ValueError('Completed chain audit failed or no longer matches raw evidence')
    return fresh


def package(destination,stem,completed_chain=None,chain_audit=None):
    if (completed_chain is None)!=(chain_audit is None):
        raise ValueError('Completed backup and its independent audit must be supplied together')
    accepted=None
    if completed_chain is not None:
        completed_chain=repository_input(completed_chain)
        chain_audit=repository_input(chain_audit)
        accepted=validate_chain_audit(
            completed_chain/'results/supplement30_common_policy_chain_resumed_724_20261009',chain_audit)
    files={}
    def add(path):
        if not path.is_file():raise FileNotFoundError(path)
        files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for folder in ('src','tests'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':add(path)
    helpers=(
        'README.md','SERVER724_DIAGNOSTIC_20261009.md','SERVER724_RANK2_STAGE_20261009.md',
        'PRESSURE_RESULTS_20261009.md','upstream_reference.py','evaluate.py',
        'exact30_acceptance.py','test_exact30_acceptance.py','profile_fixed30_bounded.py',
        'test_profile_fixed30_bounded.py','candidate_guarded_scaled.py','test_guarded_scaled.py',
        'candidate_controlled_rank2.py','test_controlled_rank2.py','candidate_controlled_adaptive.py',
        'test_controlled_adaptive.py','probe_controlled_rank2.py','test_controlled_probe.py',
        'probe_large_qr_threads.py','test_qr_thread_shapes.py','supplement30_workloads.py',
        'test_supplement30_workloads.py','paired_supplement30.py','test_paired_supplement30.py',
        'drain_stalled_supplement.py','test_drain_stalled_supplement.py','common_qr_thread_policy.py',
        'test_common_qr_policy.py','paired_supplement30_common_policy.py','test_paired_common_policy.py',
        'audit_supplement30.py','test_audit_supplement30.py','audit_common_policy_pairs.py',
        'test_audit_common_policy.py','resume_common_policy_chain.py','test_resume_common_policy_chain.py',
        'audit_resumed_common_policy_chain.py','test_audit_resumed_common_policy.py',
        'verify_stage724_backup.py','test_verify_stage724_backup.py','package_server724_stage.py',
        'test_package_completed_chain.py',
        'download_724_archive_chunks.py','test_download_724_archive_chunks.py',
        'audit_pressure_chain.py','resume_pressure_chain.py','paired_pressure_chain.py',
        'paired_upstream_mixed.py','paired_integrated_mixed.py','audit_upstream_mixed.py')
    for name in helpers:add(HERE/name)
    # Include pre-existing tracked research dependencies (for example
    # capacity_plan.py), not unrelated untracked historical experiments.
    tracked=subprocess.check_output(['git','ls-files','-z','--','benchmarks/asc_revision_cpu/*.py'],cwd=ROOT)
    for name in tracked.decode('utf-8').split('\0'):
        if name:add(ROOT/name)
    for name in ('README.md','.gitattributes','benchmarks/asc_large_cpu/benchmark.py'):add(ROOT/name)
    # Preserve unrelated user edits: use the committed project configuration.
    files['pyproject.toml']=subprocess.check_output(['git','show','HEAD:pyproject.toml'],cwd=ROOT)
    pdf=Path('C:/Users/孙逸腾/Documents/WeChat Files/wxid_mcrolp1g1yp622/FileStorage/File/2026-09/Qibotn.pdf').read_bytes()
    if sha(pdf)!='630ee41393f8c1312d76260aa705d7c6bda17dbed4a0cf242bd282874f409af9':
        raise ValueError('Authoritative requirement PDF changed')
    files['requirements/Qibotn.pdf']=pdf
    verified_files=verified_snapshots=0
    backups=[ROOT/'results/server724_revision_20261009_2056'/name
             for name in ('stage2114','common_policy_stopped2140')]
    if completed_chain is not None:backups.append(completed_chain)
    for root in backups:
        record=json.loads((root/'BACKUP_VERIFICATION.json').read_text(encoding='utf-8'))
        for name,metadata in record['members'].items():
            raw=(root/name).read_bytes()
            if sha(raw)!=metadata['sha256'] or len(raw)!=metadata['bytes']:raise ValueError('Backup changed: '+name)
        verified_files+=record['files_verified'];verified_snapshots+=record['source_snapshots_verified']
        for path in root.rglob('*'):
            if path.is_file():add(path)
    for directory in ('server724_diagnostic_20261009_1948','revision_cpu_20261009_integrated_mixed16',
                      'revision_cpu_20261009_pressure20_9991_resumed'):
        for path in (ROOT/'results'/directory).rglob('*'):
            if path.is_file():add(path)
    for name in ('pressure20_final_audit_20261009.json','audit_common_policy_stopped_20261009_2140.json'):
        add(ROOT/'results'/name)
    if chain_audit is not None:add(chain_audit)
    if accepted is not None:
        add(ROOT/'results/validation_package_completed_chain_20261009.xml')
        add(ROOT/'results/validation_package_completed_chain_compressed_20261009.xml')
        add(ROOT/'results/validation_package_completed_chain_final_20261009.xml')
        add(completed_chain.parent/'qibotn_chain_resumed_complete_20261009_2254_chunked.tar.gz.transfer.json')
    # Include success AND initial failed/resource-protected test receipts.
    patterns=('*common*20261009.xml','*supplement30*20261009.xml','*controlled*20261009.xml',
              '*724*20261009.xml','*stage724*20261009.xml')
    for pattern in patterns:
        for path in (ROOT/'results').glob(pattern):add(path)
    preflight=ROOT/'results/server724_revision_20261009_2056/resume_preflight2152'
    for path in preflight.rglob('*'):
        if path.is_file():add(path)
    test_path=preflight/'validation_resume_and_audit_common_policy_724_v2_20261009.xml'
    suites=ET.parse(test_path).getroot().findall('testsuite')
    totals={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    if totals!={'tests':35,'failures':0,'errors':0,'skipped':0}:raise ValueError('Final server preflight tests differ')
    delivery_tests=None
    if accepted is not None:
        suites=ET.parse(ROOT/'results/validation_package_completed_chain_final_20261009.xml').getroot().findall('testsuite')
        delivery_tests={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
        if delivery_tests!={'tests':32,'failures':0,'errors':0,'skipped':0}:raise ValueError('Delivery guard tests differ')
    receipt={'status':'verified_stage_backup_not_final_acceptance','server_preflight_tests':totals,
             'delivery_guard_tests':delivery_tests,
             'evidence_files_verified':verified_files,'source_snapshots_verified':verified_snapshots,
             'original30_complete':False,'supplement30_suite_complete':False,'current_30q_speedup':None,
             'word_pdf_updated':False,'submission_sent':False,
             'live_run_results_included':False,'unrelated_pyproject_edits_preserved':True,
             'configuration_bytes':'committed HEAD:pyproject.toml, not the unrelated dirty working file'}
    receipt.update(formal_chain_supplement_complete=accepted is not None,
                   supplementary_chain_speedups_vs_official_common_qr_policy=(
                       accepted['chain']['speedups_vs_official_common_qr_policy'] if accepted else None),
                   supplementary_scope='one predeclared six-bridge chain only; not original high-entanglement30')
    files['STAGE_RECEIPT.json']=json.dumps(receipt,ensure_ascii=False,indent=2).encode('utf-8')
    manifest={name:{'bytes':len(raw),'sha256':sha(raw)} for name,raw in sorted(files.items())}
    manifest_bytes=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    destination.mkdir(parents=True,exist_ok=True)
    archive_path=destination/(stem+'.zip');report_path=destination/(stem+'.md')
    receipt_path=destination/(stem+'_交付核验.json');sha_path=destination/(stem+'.zip.sha256')
    if any(p.exists() for p in (archive_path,report_path,receipt_path,sha_path)):
        raise FileExistsError('Refusing to replace an old delivery')
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):archive.writestr(name,raw)
        archive.writestr('MANIFEST_SHA256.json',manifest_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(files)|{'MANIFEST_SHA256.json'} or archive.testzip() is not None:
            raise ValueError('ZIP membership/CRC failure')
        if archive.read('MANIFEST_SHA256.json')!=manifest_bytes:raise ValueError('Manifest changed')
        for name,raw in files.items():
            if sha(archive.read(name))!=sha(raw):raise ValueError('ZIP byte hash mismatch: '+name)
    shutil.copyfile(HERE/'SERVER724_RANK2_STAGE_20261009.md',report_path)
    if sha(report_path.read_bytes())!=sha((HERE/'SERVER724_RANK2_STAGE_20261009.md').read_bytes()):raise ValueError('Report copy changed')
    receipt.update(zip_sha256=sha(archive_path.read_bytes()),member_hashes_verified=len(files),
                   members_including_manifest=len(files)+1,crc_verified=True,archive=str(archive_path),report=str(report_path))
    with receipt_path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    with sha_path.open('x',encoding='utf-8') as stream:stream.write(receipt['zip_sha256']+'  '+archive_path.name+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('C:/Users/孙逸腾/Desktop/超算'))
    parser.add_argument('--stem',default='QiboTN_724精确30比特阶段备份_20261009_2159')
    parser.add_argument('--completed-chain',type=Path)
    parser.add_argument('--chain-audit',type=Path)
    args=parser.parse_args()
    print(json.dumps(package(args.destination,args.stem,args.completed_chain,args.chain_audit),ensure_ascii=False,indent=2))
