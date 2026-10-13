import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    name='posttrain.training.verl_pilot'
    assert importlib.util.find_spec(name) is not None,'pilot entry point not implemented'
    return importlib.import_module(name)


def test_pilot_cannot_be_relabeled_formal_or_shrink_stable_measurement():
    module=implementation()
    cfg=json.loads(Path('configs/verl_pilot.json').read_text(encoding='utf-8'))
    module.checked_config(cfg)
    for patch in ({'scope':'formal_training'},{'formal_training_supported':True},
                  {'total_steps':3},{'model_id':'Qwen/1.5B'},
                  {'reference_policy':'unrecorded'},{'g':1},
                  {'token_cap':1},{'dtype':'float16'}):
        with pytest.raises(ValueError):module.checked_config({**cfg,**patch})


def test_pilot_dataset_identity_requires_matching_parquet_and_prompt_sources(tmp_path):
    module=implementation()
    from posttrain.common import file_digest,write_json
    dataset=tmp_path/'dataset';dataset.mkdir()
    sources=[]
    for split in ('train','development'):
        path=tmp_path/(split+'.jsonl');path.write_text('fixture',encoding='utf-8')
        sources.append({'path':str(path),'sha256':file_digest(path)})
        (dataset/(split+'.parquet')).write_bytes(b'fixture parquet')
    manifest={'scope':'engineering_pilot_only','verl_revision':module.VERL_REVISION,
        'source_manifest_sha256':dict(zip(('train','development'),(s['sha256'] for s in sources))),
        'parquet_sha256':{s:file_digest(dataset/(s+'.parquet')) for s in ('train','development')},
        'train_prompts':51,'development_prompts':9}
    write_json(dataset/'manifest.json',manifest)
    module.checked_dataset(dataset,sources)
    (dataset/'train.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='parquet'):
        module.checked_dataset(dataset,sources)


def test_pilot_rejects_missing_or_altered_model_shard(tmp_path):
    module=implementation()
    from posttrain.common import file_digest
    shard=tmp_path/'model-00001-of-00001.safetensors';shard.write_bytes(b'fixture')
    manifest={'model_id':module.MODEL,'revision':module.MODEL_REVISION,
        'files':{shard.name:{'bytes':7,'sha256':file_digest(shard)}}}
    module.checked_model_files(tmp_path,manifest)
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError,match='model_file'):
        module.checked_model_files(tmp_path,manifest)


def test_model_index_cannot_point_to_unverified_extra_weights(tmp_path):
    module=implementation()
    from posttrain.common import file_digest,write_json
    shard=tmp_path/'model-00001-of-00001.safetensors';shard.write_bytes(b'fixture')
    index=tmp_path/'model.safetensors.index.json'
    write_json(index,{'weight_map':{'layer.weight':'unverified.safetensors'}})
    manifest={'model_id':module.MODEL,'revision':module.MODEL_REVISION,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in (shard,index)}}
    with pytest.raises(ValueError,match='model_index'):
        module.checked_model_files(tmp_path,manifest)
