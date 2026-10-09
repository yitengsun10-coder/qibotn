"""Verify and safely unpack a copied evidence TAR, preserving every raw byte.

This checks transport, membership and declared source snapshots. It is NOT a
correctness or performance acceptance audit and never promotes a 30q result.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_members(members):
    names = set()
    files = []
    total = 0
    for member in members:
        name = member.name
        path = PurePosixPath(name)
        if (path.is_absolute() or not path.parts or '..' in path.parts or
                '\\' in name or ':' in name or path.parts[0] != 'results'):
            raise ValueError('Unsafe evidence member: ' + name)
        normalized = path.as_posix().rstrip('/')
        if normalized in names:
            raise ValueError('Duplicate evidence member: ' + name)
        names.add(normalized)
        if not (member.isdir() or member.isfile()):
            raise ValueError('Links/special members are forbidden: ' + name)
        if member.isfile():
            total += member.size
            files.append(member)
    if total > 256 * 2**20:
        raise ValueError('Evidence exceeds explicit 256-MiB unpack budget')
    return files


def verify_snapshots(root):
    verified = []
    for path in sorted(root.rglob('summary.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'source_hashes' not in record:
            continue
        for name, expected in record['source_hashes'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe source snapshot declaration: ' + name)
            snapshot = path.parent / 'source_snapshot' / name
            actual = sha256(snapshot.read_bytes())
            if actual != expected:
                raise ValueError('Source snapshot differs: ' + str(snapshot))
            verified.append({'path': snapshot.relative_to(root).as_posix(), 'sha256': actual})
    return verified


def unpack(archive_path, expected_sha, destination):
    if destination.exists():
        raise FileExistsError('Fresh evidence directory required')
    raw = archive_path.read_bytes()
    actual = sha256(raw)
    if actual != expected_sha.lower():
        raise ValueError('Transport SHA256 differs')
    # Validate the entire inventory before creating the output directory.
    with tarfile.open(archive_path, 'r:') as archive:
        files = validate_members(archive.getmembers())
        destination.mkdir(parents=True)
        inventory = {}
        for member in files:
            data = archive.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError('Short TAR member: ' + member.name)
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            digest = sha256(data)
            if sha256(target.read_bytes()) != digest:
                raise ValueError('Unpacked bytes differ: ' + member.name)
            inventory[member.name] = {'bytes': len(data), 'sha256': digest}
    snapshots = verify_snapshots(destination)
    receipt = {
        'status': 'transport_and_raw_snapshot_verification_only',
        'archive_sha256': actual,
        'archive_bytes': len(raw),
        'files_verified': len(inventory),
        'source_snapshots_verified': len(snapshots),
        'members': inventory,
        'source_snapshots': snapshots,
        'original30_complete': False,
        'all_contest_tasks_complete': False,
        'performance_acceptance_audit_performed': False,
    }
    with (destination / 'BACKUP_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    return {key: value for key, value in receipt.items() if key not in ('members', 'source_snapshots')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.sha256, args.output), ensure_ascii=False, indent=2))
