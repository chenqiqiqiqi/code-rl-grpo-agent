from dataclasses import replace
import json

import pytest

from posttrain.common import write_jsonl
from posttrain.data.prompt_manifest import public_prompt
from posttrain.training.verl_data import export_pilot,pilot_rows


def examples(problem):
    train=replace(problem,validation_status='reference_passed',cluster_id='train-cluster')
    development=replace(train,problem_id='development',cluster_id='dev-cluster')
    return [public_prompt(train,'train')],[public_prompt(development,'development')]


def test_verl_reward_input_contains_identifiers_not_hidden_tests(problem):
    train,development=examples(problem)
    rows,_=pilot_rows(train,development)
    assert rows[0]['prompt']==train[0]['messages']
    assert json.loads(rows[0]['reward_model']['ground_truth'])=={
        'problem_id':problem.problem_id,'tests_hash':problem.tests_hash,'test_count':len(problem.tests)}
    assert 'references' not in rows[0] and 'tests' not in rows[0]
    with pytest.raises(ValueError,match='unexpected_GPU_input_fields'):
        pilot_rows([{**train[0],'hidden_tests':[{'input':'private'}]}],development)


def test_verl_export_rejects_final_tasks_and_cross_split_clusters(problem):
    train,development=examples(problem)
    with pytest.raises(ValueError,match='preparation_split_only'):
        pilot_rows([{**train[0],'split':'final'}],development)
    with pytest.raises(ValueError,match='cluster_overlap'):
        pilot_rows(train,[{**development[0],'cluster_id':train[0]['cluster_id']}])
    with pytest.raises(ValueError,match='duplicate_prompt_id'):
        pilot_rows(train*2,development)


def test_verl_parquet_roundtrip_is_static_evidence(tmp_path,problem):
    pytest.importorskip('pyarrow')
    train,development=examples(problem)
    first=tmp_path/'train.jsonl';second=tmp_path/'dev.jsonl'
    write_jsonl(first,train);write_jsonl(second,development)
    report=export_pilot(first,second,tmp_path/'output')
    assert report['train_prompts']==report['development_prompts']==1
    assert not report['formal_training_eligible'] and not report['tokenizer_length_checked']
    assert report['model_updates']==0 and not report['hidden_tests_exported']
    with pytest.raises(ValueError,match='immutable_dataset_output'):
        export_pilot(first,second,tmp_path/'output')
