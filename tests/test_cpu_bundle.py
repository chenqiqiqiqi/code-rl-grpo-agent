import hashlib
import importlib.util
import json
import io
import tarfile
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('verify_cpu_bundle',Path(__file__).resolve().parents[1]/'scripts/verify_cpu_bundle.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def bundle(tmp_path):
    (tmp_path/'source.py').write_text('print(1)',encoding='utf-8')
    manifest={'bundle_id':'synthetic-contract','includes_credentials':False,
        'files':{'source.py':hashlib.sha256(b'print(1)').hexdigest()}}
    (tmp_path/'bundle-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


def test_complete_bundle_hashes(tmp_path):
    bundle(tmp_path)
    assert module.verify(tmp_path)['checked_files']==1


def test_changed_code_refused(tmp_path):
    bundle(tmp_path);(tmp_path/'source.py').write_text('print(2)',encoding='utf-8')
    with pytest.raises(ValueError,match='bundle_file_changed'):module.verify(tmp_path)


def test_extra_unregistered_code_refused(tmp_path):
    bundle(tmp_path);(tmp_path/'sitecustomize.py').write_text('pass',encoding='utf-8')
    with pytest.raises(ValueError,match='unexpected_bundle_files'):module.verify(tmp_path)


def test_path_outside_bundle_refused(tmp_path):
    manifest=bundle(tmp_path);manifest['files']['../outside.py']='0'*64
    (tmp_path/'bundle-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='unsafe_bundle_path'):module.verify(tmp_path)


def test_oci_manifest_identity_bound_to_original_config(tmp_path):
    config=b'{"architecture":"amd64"}'
    config_id='sha256:'+hashlib.sha256(config).hexdigest()
    manifest=json.dumps({'config':{'digest':config_id}}).encode()
    manifest_id='sha256:'+hashlib.sha256(manifest).hexdigest()
    archive=tmp_path/'image.tar.gz'
    entries={'index.json':json.dumps({'manifests':[{'digest':manifest_id}]}).encode(),
             'blobs/sha256/'+manifest_id[7:]:manifest,'blobs/sha256/'+config_id[7:]:config}
    with tarfile.open(archive,'w:gz') as t:
        for name,data in entries.items():
            info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    assert module.image_identities(archive,config_id)==[config_id,manifest_id]
    with pytest.raises(ValueError,match='image_config_identity_mismatch'):
        module.image_identities(archive,'sha256:'+'0'*64)
