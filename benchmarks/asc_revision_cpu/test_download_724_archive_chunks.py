"""Synthetic transport checks; no SSH or performance measurement."""
import base64
import hashlib
import pytest
from benchmarks.asc_revision_cpu.download_724_archive_chunks import decode_chunk,assemble_chunks,download


def test_chunk_and_whole_hash():
    raw=b'raw\r\nbytes\x00'
    assert decode_chunk(base64.b64encode(raw),len(raw))==raw
    assert assemble_chunks([raw[:4],raw[4:]],len(raw),hashlib.sha256(raw).hexdigest())==raw


@pytest.mark.parametrize('encoded,size',[(b'not base64',1),(b'YQ==',2)])
def test_reject_bad_chunk(encoded,size):
    with pytest.raises(ValueError):decode_chunk(encoded,size)


def test_reject_bad_assembly_and_overwrite(tmp_path):
    with pytest.raises(ValueError):assemble_chunks([b'a'],1,'0'*64)
    output=tmp_path/'old.tar.gz';output.write_bytes(b'old')
    with pytest.raises(FileExistsError):download(output,1,'0'*64)
    assert output.read_bytes()==b'old'
