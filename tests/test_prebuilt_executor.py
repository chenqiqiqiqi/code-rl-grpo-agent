import importlib.util
import json
from pathlib import Path

import pytest

from posttrain.common import file_digest,write_json


spec=importlib.util.spec_from_file_location('verify_bundle',Path(__file__).resolve().parents[1]/'scripts/verify_cpu_bundle.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup(root):
    root.mkdir();(root/'docker').mkdir();(root/'manifests').mkdir()
    worker=root/'docker/worker.py';worker.write_text('# trusted worker')
    image=root.parent/'image.tar.gz';image.write_bytes(b'synthetic_hash_fixture_not_a_real_image')
    identity='sha256:'+'1'*64
    write_json(root/'manifests/executor_image_v3_portable.json',{
        'archive_sha256':file_digest(image),'archive_bytes':image.stat().st_size,
        'image_id':identity,'worker_source_sha256':file_digest(worker)})
    return image,identity


def verifier():
    assert hasattr(module,'verify_prebuilt_image'),'prebuilt executor verification not implemented'
    return module.verify_prebuilt_image


def test_prebuilt_archive_is_bound_to_current_worker_and_digest(tmp_path):
    archive,identity=setup(tmp_path/'bundle')
    assert verifier()(tmp_path/'bundle',archive)==identity
    archive.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive_hash'):
        verifier()(tmp_path/'bundle',archive)


def test_prebuilt_archive_with_different_worker_is_rejected(tmp_path):
    archive,_=setup(tmp_path/'bundle')
    (tmp_path/'bundle/docker/worker.py').write_text('# different worker')
    with pytest.raises(ValueError,match='worker_source'):
        verifier()(tmp_path/'bundle',archive)


def test_prebuilt_image_mutable_tag_is_rejected(tmp_path):
    archive,_=setup(tmp_path/'bundle')
    manifest=tmp_path/'bundle/manifests/executor_image_v3_portable.json'
    row=json.loads(manifest.read_text(encoding='utf-8'));row['image_id']='untrusted:latest';write_json(manifest,row)
    with pytest.raises(ValueError,match='immutable_image_id'):
        verifier()(tmp_path/'bundle',archive)
