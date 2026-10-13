from dataclasses import replace
import pytest
from posttrain.training.sft import checked_config,checked_examples,update_plan


def test_sft_config_cannot_change_model_or_use_quantization():
    cfg={'model_id':'Qwen/Qwen2.5-Coder-7B-Instruct','model_revision':'a'*40,
         'max_steps':20,'batch_size':4,'max_length':1024,'seed':42,'learning_rate':2e-5,
         'lora_rank':32,'lora_alpha':64,'save_steps':[5,10,15,20],'dtype':'bfloat16'}
    assert checked_config(cfg)==cfg
    with pytest.raises(ValueError): checked_config({**cfg,'model_id':'smaller-model'})
    with pytest.raises(ValueError): checked_config({**cfg,'dtype':'int4'})
    with pytest.raises(ValueError): checked_config({**cfg,'save_steps':[5,10]})


def test_sft_manifest_requires_valid_references_and_disjoint_clusters(problem):
    valid=replace(problem,validation_status='reference_passed',cluster_id='train')
    dev=replace(valid,problem_id='dev',cluster_id='dev')
    checked_examples([valid],[dev])
    with pytest.raises(ValueError):checked_examples([valid],[replace(dev,cluster_id='train')])
    with pytest.raises(ValueError):checked_examples([problem],[dev])


def test_update_order_is_reproducible_for_resume_and_never_partial():
    full=update_plan(5,3,7,42)
    assert len(full)==7 and all(len(x)==3 for x in full)
    assert full==update_plan(5,3,7,42)
    assert full!=update_plan(5,3,7,43)
    assert full[3:]==update_plan(5,3,7,42)[3:]


def test_resume_detects_swapped_adapter_files(tmp_path):
    from posttrain.training.sft import checked_checkpoint
    from posttrain.common import file_digest,write_json
    (tmp_path/'adapter').mkdir()
    weights=tmp_path/'adapter/adapter_model.safetensors';weights.write_bytes(b'original')
    state=tmp_path/'training_state.pt';state.write_bytes(b'state')
    write_json(tmp_path/'state_manifest.json',{'step':5,'state_sha256':file_digest(state),
        'adapter_sha256':{weights.name:file_digest(weights)}})
    assert checked_checkpoint(tmp_path)['step']==5
    weights.write_bytes(b'swapped')
    with pytest.raises(ValueError,match='adapter_checksum_mismatch'):checked_checkpoint(tmp_path)


def test_resume_identity_binds_exact_encoded_tokens_and_loss_mask(tmp_path):
    from posttrain.training.sft import input_identity
    examples=[('p',{'input_ids':[1,2,3],'labels':[-100,2,3],'attention_mask':[1,1,1]})]
    baseline=input_identity(examples,{'torch':'test-version'})
    assert baseline==input_identity(examples,{'torch':'test-version'})
    changed=[('p',{**examples[0][1],'labels':[-100,-100,3]})]
    assert baseline!=input_identity(changed,{'torch':'test-version'})
    assert baseline!=input_identity(examples,{'torch':'different-version'})


def test_resume_rejects_stale_complete_tail_and_ignores_temporary_save(tmp_path):
    from posttrain.training.sft import require_latest_checkpoint
    first=tmp_path/'checkpoint-000005';first.mkdir()
    (tmp_path/'.checkpoint-000010-incomplete-crash').mkdir()
    require_latest_checkpoint(tmp_path,first)
    latest=tmp_path/'checkpoint-000010';latest.mkdir()
    with pytest.raises(ValueError,match='resume_latest_complete_checkpoint'):
        require_latest_checkpoint(tmp_path,first)
    require_latest_checkpoint(tmp_path,latest)


def test_adapter_origin_must_match_generation_base_revision(tmp_path):
    from posttrain.common import digest,file_digest,write_json
    from posttrain.training.generate import checked_adapter_origin
    checkpoint=tmp_path/'checkpoint-000001';(checkpoint/'adapter').mkdir(parents=True)
    (checkpoint/'adapter/adapter_model.safetensors').write_bytes(b'weights')
    (checkpoint/'training_state.pt').write_bytes(b'state')
    identity={'config':{'model_id':'Qwen/Qwen2.5-Coder-7B-Instruct','model_revision':'a'*40}}
    write_json(tmp_path/'identity.json',identity)
    write_json(checkpoint/'state_manifest.json',{'state_sha256':file_digest(checkpoint/'training_state.pt'),
        'training_identity_sha256':digest(identity),'adapter_sha256':{
            'adapter_model.safetensors':file_digest(checkpoint/'adapter/adapter_model.safetensors')}})
    checked_adapter_origin(checkpoint,identity['config'])
    with pytest.raises(ValueError,match='adapter_base_revision_mismatch'):
        checked_adapter_origin(checkpoint,{**identity['config'],'model_revision':'b'*40})
