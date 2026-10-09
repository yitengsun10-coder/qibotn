"""Synthetic safety/transport fixtures; not performance data."""
import hashlib
import io
import json
import tarfile
import pytest
from benchmarks.asc_revision_cpu.verify_stage724_backup import unpack, validate_members


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'results/../escape',
                                  'C:/escape', 'results\\escape', 'other/file'])
def test_reject_unsafe_names(name):
    with pytest.raises(ValueError):
        validate_members([tarfile.TarInfo(name)])


def test_reject_link_and_duplicate():
    member = tarfile.TarInfo('results/link'); member.type = tarfile.SYMTYPE
    with pytest.raises(ValueError):
        validate_members([member])
    with pytest.raises(ValueError):
        validate_members([tarfile.TarInfo('results/a'), tarfile.TarInfo('results/a')])


@pytest.mark.parametrize('mode,suffix', [('w', '.tar'), ('w:gz', '.tar.gz')])
def test_transport_and_snapshot_checks(tmp_path, mode, suffix):
    source = b'# raw source\r\n'
    summary = json.dumps({'source_hashes': {'test.py': hashlib.sha256(source).hexdigest()}}).encode()
    path = tmp_path / ('evidence' + suffix)
    with tarfile.open(path, mode) as archive:
        for name, data in [('results/case/summary.json', summary),
                           ('results/case/source_snapshot/test.py', source)]:
            member = tarfile.TarInfo(name); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        unpack(path, '0' * 64, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    output = tmp_path / 'good'
    receipt = unpack(path, digest, output)
    assert receipt['files_verified'] == 2 and receipt['source_snapshots_verified'] == 1
    assert not receipt['original30_complete']
    assert (output / 'results/case/source_snapshot/test.py').read_bytes() == source
    with pytest.raises(FileExistsError):
        unpack(path, digest, output)
