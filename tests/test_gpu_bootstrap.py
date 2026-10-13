import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

from posttrain.common import file_digest, write_json


def load_bootstrap():
    spec=importlib.util.spec_from_file_location('bootstrap_repair',Path(__file__).resolve().parents[1]/'scripts/bootstrap_gpu.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_compile_rejects_small_container_even_when_host_has_large_ram(tmp_path):
    module=load_bootstrap()
    (tmp_path/'memory.max').write_text(str(2*1024**3))
    with pytest.raises(RuntimeError,match='container_memory'):
        module.check_build_memory(tmp_path)


def test_compile_allows_sufficient_container_memory(tmp_path):
    module=load_bootstrap()
    (tmp_path/'memory.max').write_text(str(120*1024**3))
    module.check_build_memory(tmp_path)


def test_source_artifact_rejects_corrupted_cache_before_execution(tmp_path):
    module=load_bootstrap()
    (tmp_path/'flash.tar.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact_checksum'):
        module.fetch_artifact({'filename':'flash.tar.gz','url':'https://invalid.example/source',
            'sha256':'0'*64},tmp_path)


def test_source_build_forces_local_build_and_preserves_dependency_versions(tmp_path,monkeypatch):
    module=load_bootstrap();calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda cmd,**kw:calls.append((cmd,kw)))
    module.install_flash_source(tmp_path/'python',tmp_path/'flash.tar.gz',tmp_path/'wheel.whl',tmp_path)
    assert len(calls)==3
    build,options=calls[1]
    assert '--no-deps' in build and '--no-build-isolation' in build
    assert '--no-cache' in build and '--no-index' in build
    assert options['env']['FLASH_ATTENTION_FORCE_BUILD']=='TRUE'
    assert options['env']['FLASH_ATTN_CUDA_ARCHS']=='80'
    assert options['timeout']==1800
    assert calls[2][0][-2:]==['pip','check'] or 'check' in calls[2][0]


def test_source_build_failure_stops_before_dependency_acceptance(tmp_path,monkeypatch):
    module=load_bootstrap();calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        if '--no-build-isolation' in cmd:raise module.subprocess.CalledProcessError(1,cmd)
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(module.subprocess.CalledProcessError):
        module.install_flash_source(tmp_path/'python',tmp_path/'flash.tar.gz',tmp_path/'wheel.whl',tmp_path)
    assert len(calls)==2


@pytest.mark.parametrize('change',['modify','remove','extra','bytecode'])
def test_existing_gpu_source_cannot_claim_verified_after_code_change(tmp_path,monkeypatch,change):
    script=Path(__file__).resolve().parents[1]/'scripts/bootstrap_gpu.py'
    spec=importlib.util.spec_from_file_location('bootstrap_gpu_under_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'project';runtime=tmp_path/'runtime';runtime.mkdir()
    checkout=runtime/'verl-synthetic'; checkout.mkdir()
    files={name:b'pinned fixture\n' for name in ('uv.lock','pyproject.toml','setup.py','manage_envs.py','verl/__init__.py')}
    archive=runtime/'verl-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,data in files.items():
            member=tarfile.TarInfo('verl-synthetic/'+name);member.size=len(data)
            tar.addfile(member,io.BytesIO(data))
            dest=checkout/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            if '/' not in name:
                dest=root/'environments/verl-0.9.1'/name
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    write_json(root/'manifests/verl_source.json',{'revision':'synthetic','archive_sha256':file_digest(archive)})
    if change=='modify':(checkout/'verl/__init__.py').write_text('modified')
    elif change=='remove':(checkout/'verl/__init__.py').unlink()
    elif change=='extra':(checkout/'verl/injected.py').write_text('extra')
    else:
        bytecode=checkout/'verl/__pycache__/__init__.cpython-313.pyc'
        bytecode.parent.mkdir();bytecode.write_bytes(b'untrusted_cached_code_fixture')
    monkeypatch.setattr(module,'ROOT',root)
    monkeypatch.setattr(module.sys,'argv',['bootstrap','--runtime-root',str(runtime),'--source-only'])
    if change=='bytecode':
        module.main()
        assert not bytecode.exists()
    else:
        with pytest.raises(ValueError,match='upstream_source'):module.main()
