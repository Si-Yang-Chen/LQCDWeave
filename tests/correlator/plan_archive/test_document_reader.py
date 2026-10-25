"""Document loading retains gzip magic, text, and strict JSON behavior."""
import gzip
import json
import pytest

from lqcdweave.storage.contraction_plans import api


@pytest.mark.parametrize('compressed', (False, True))
@pytest.mark.parametrize('bom', (False, True))
@pytest.mark.parametrize('kind', ('bytes', 'path', 'str-path'))
def test_document_reader_preserves_magic_bom_and_input_forms(tmp_path, compressed, bom, kind):
    document = {'name': '关联函数 α', 'fields': [1, -2, None, True], 'line': 'a\r\nb'}
    raw = json.dumps(document, ensure_ascii=False).encode('utf-8')
    if bom:
        raw = b'\xef\xbb\xbf' + raw
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    # Compression is selected by magic, independently of the filename suffix.
    path = tmp_path / ('plan.json' if compressed else 'plan.json.gz')
    path.write_bytes(raw)
    source = raw if kind == 'bytes' else path if kind == 'path' else str(path)
    assert api._read_document(source) == document


def test_document_reader_keeps_inline_text(tmp_path):
    assert api._read_document(' \n{"name":"关联函数"}') == {'name': '关联函数'}
    with pytest.raises(api.PlanArchiveError, match='unable to read'):
        api._read_document(tmp_path / 'missing.json')
    with pytest.raises(TypeError, match='archive source'):
        api._read_document({})


@pytest.mark.parametrize('raw', (
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
    b'{"x":"\xff"}', b'{"x":1} trailing',
))
@pytest.mark.parametrize('compressed', (False, True))
def test_document_reader_keeps_strict_json_errors(raw, compressed):
    data = gzip.compress(raw, mtime=0) if compressed else raw
    with pytest.raises(api.PlanArchiveError, match='invalid plan archive JSON'):
        api._read_document(data)


def test_document_reader_rejects_truncated_crc_and_gzip_trailing_data():
    raw = gzip.compress(b'{"x":1}', mtime=0)
    bad_crc = bytearray(raw)
    bad_crc[-8] ^= 1
    for data in (raw[:-5], bytes(bad_crc), raw + b'not a gzip member'):
        with pytest.raises(api.PlanArchiveError, match='invalid compressed plan archive'):
            api._read_document(data)
    # Concatenated members remain supported by the original gzip container.
    assert api._read_document(gzip.compress(b'{"x":', mtime=0) +
                              gzip.compress(b'1}', mtime=0)) == {'x': 1}
