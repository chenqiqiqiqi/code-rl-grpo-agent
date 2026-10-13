from dataclasses import replace
import importlib

import pytest

from posttrain.common import digest,read_jsonl,write_json,write_jsonl
from posttrain.execution.contracts import CaseResult,JudgeResult,Status


def test_reference_partition_covers_every_problem_once_and_is_frozen(tmp_path,problem):
    module=importlib.import_module('posttrain.data.reference_pool')
    source=tmp_path/'input.jsonl'
    rows=[replace(problem,problem_id=f'p{i}',statement=f'synthetic statement {i}').to_dict() for i in range(11)]
    write_jsonl(source,rows)
    manifest=module.prepare(source,tmp_path/'pool',workers=4,max_references=3)
    actual=[]
    for shard in manifest['shards']:
        actual.extend(read_jsonl(tmp_path/'pool'/shard['input']))
    assert sorted(x['problem_id'] for x in actual)==sorted(x['problem_id'] for x in rows)
    assert max(s['problems'] for s in manifest['shards'])==3
    with pytest.raises(FileExistsError):module.prepare(source,tmp_path/'pool',workers=4,max_references=3)


def test_incomplete_shard_and_changed_input_prevent_merge(tmp_path,problem):
    module=importlib.import_module('posttrain.data.reference_pool')
    source=tmp_path/'input.jsonl'
    write_jsonl(source,[replace(problem,problem_id=f'p{i}').to_dict() for i in range(4)])
    output=tmp_path/'pool'; manifest=module.prepare(source,output,workers=2,max_references=3)
    with pytest.raises(ValueError,match='incomplete_shard'):module.collect(output)
    shard=manifest['shards'][0];(output/shard['input']).write_text('{}\n')
    with pytest.raises(ValueError,match='input_changed'):module.checked_manifest(output)


def completed_fixture(tmp_path,problem):
    module=importlib.import_module('posttrain.data.reference_pool')
    source=tmp_path/'input.jsonl';write_jsonl(source,[problem.to_dict()])
    output=tmp_path/'pool';manifest=module.prepare(source,output,workers=1,max_references=3)
    shard=manifest['shards'][0];run=output/shard['run']
    result=JudgeResult('synthetic-result',Status.OK,5,5,
        tuple(CaseResult(Status.OK,stdout=t.stdout) for t in problem.tests),.1,0,5,0,False,
        'test-fixture-not-live','isolated_execution',digest(problem.references[0]),problem.tests_hash)
    write_json(run/'run_manifest.json',{'identity':{'source_sha256':shard['sha256'],
        'max_references':3,'executor_version':result.executor_version,'scoring_code_sha256':'fixture'}})
    write_json(run/'status.json',{'complete':True,'reference_passed':1,'processed':1,
        'physical_executions':5,'known_physical_executions':5,'unknown_physical_attempts':0})
    write_json(run/'records/00000-001.json',{'complete':True,'problem_id':problem.problem_id,
        'physical_accounting_complete':True,'accepted':replace(problem,validation_status='reference_passed').to_dict(),
        'audit':[{'problem_id':problem.problem_id,'reference_index':0,'result':result.to_dict()}]})
    return module,output,run


def test_collection_success_binds_record_files(tmp_path,problem):
    module,output,run=completed_fixture(tmp_path,problem)
    result=module.collect(output)
    assert result['reference_passed']==1 and result['physical_executions']==5
    assert result['shards'][0]['record_sha256']


def test_status_cannot_erase_recorded_physical_work(tmp_path,problem):
    module,output,run=completed_fixture(tmp_path,problem)
    write_json(run/'status.json',{'complete':True,'reference_passed':1,'processed':1,
        'physical_executions':0,'known_physical_executions':0,'unknown_physical_attempts':0})
    with pytest.raises(ValueError,match='physical_accounting'):module.collect(output)
