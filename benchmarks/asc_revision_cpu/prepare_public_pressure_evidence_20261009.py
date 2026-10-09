"""Copy only reviewed raw benchmark evidence for the user's authorized fork.

Does not copy the private requirements PDF, ZIP, unrelated workspace files,
credentials or virtual environment. Does not stage, commit or push itself.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    from benchmarks.asc_revision_cpu.audit_pressure_chain import audit as pressure_audit
    from benchmarks.asc_revision_cpu.audit_upstream_mixed import audit as mixed_audit
    from benchmarks.asc_revision_cpu.package_pressure_20261009 import DATASETS
    pressure_audit(ROOT / 'results/revision_cpu_20261009_pressure20_9991_resumed')
    mixed_audit(ROOT / 'results/revision_cpu_20261009_integrated_mixed16')
    destination = HERE / 'evidence_20261009'
    if destination.exists():
        raise FileExistsError('Do not overwrite published evidence staging')
    destination.mkdir()
    originals = []
    for dataset in DATASETS:
        source = ROOT / ('results/revision_cpu_20261009_' + dataset)
        target = destination / source.name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        originals.extend((path, target / path.relative_to(source)) for path in source.rglob('*')
                         if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
    for name in ('pressure20_final_audit_20261009.json', 'validation_exact30_preparation_20261009.xml'):
        source, target = ROOT / 'results' / name, destination / name
        shutil.copyfile(source, target)
        originals.append((source, target))
    members = {}
    for source, target in originals:
        before, after = source.read_bytes(), target.read_bytes()
        if before != after:
            raise ValueError('Public evidence copy differs: ' + str(target))
        members[target.relative_to(destination).as_posix()] = {
            'sha256': hashlib.sha256(after).hexdigest(), 'bytes': len(after)}
    receipt = {'status': 'verified_stage_not_all_done', 'members': members,
        'formal_30q_acceptance_complete': False, 'seven_x_achieved': False,
        'note': 'Original time stop and restart warmups preserved. Requirements PDF is local-only.'}
    (destination / 'MANIFEST_SHA256.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'prepared_verified_public_evidence', 'files': len(members),
                      'directory': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
