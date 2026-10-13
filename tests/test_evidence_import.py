import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

from posttrain.common import file_digest


spec=importlib.util.spec_from_file_location('import_runs',Path(__file__).resolve().parents[1]/'scripts/import_run_archive.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def make_zip(path,entries):
    with ZipFile(path,'w') as archive:
        for name,data in entries:archive.writestr(name,data)


def test_run_evidence_hash_and_safe_import(tmp_path):
    archive=tmp_path/'evidence.zip';make_zip(archive,[('runs/one/result.json','original')])
    with pytest.raises(ValueError,match='hash_mismatch'):
        module.import_archive(archive,'wrong',tmp_path,tmp_path/'receipt.json')
    assert not (tmp_path/'runs').exists()
    receipt=module.import_archive(archive,file_digest(archive),tmp_path,tmp_path/'receipt.json')
    assert receipt['new_files']==1
    receipt=module.import_archive(archive,file_digest(archive),tmp_path,tmp_path/'receipt2.json')
    assert receipt['identical_existing_files']==1


@pytest.mark.parametrize('name',['runs/../../escape','runs/one/file:stream','outside/result'])
def test_run_evidence_rejects_traversal_before_writing(tmp_path,name):
    archive=tmp_path/'evidence.zip';make_zip(archive,[('runs/ok/result','a'),(name,'b')])
    with pytest.raises(ValueError,match='unsafe_archive_path'):
        module.import_archive(archive,file_digest(archive),tmp_path,tmp_path/'receipt.json')
    assert not (tmp_path/'runs').exists()


def test_run_evidence_preserves_conflicting_existing_file(tmp_path):
    target=tmp_path/'runs/one/result.json';target.parent.mkdir(parents=True);target.write_text('keep')
    archive=tmp_path/'evidence.zip';make_zip(archive,[('runs/one/result.json','different')])
    with pytest.raises(ValueError,match='existing_evidence_differs'):
        module.import_archive(archive,file_digest(archive),tmp_path,tmp_path/'receipt.json')
    assert target.read_text()=='keep'
