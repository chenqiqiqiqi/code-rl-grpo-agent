from dataclasses import replace
import importlib
import importlib.util
import json

import pytest

from posttrain.common import digest
from posttrain.data.prompt_manifest import public_prompt
from posttrain.execution.contracts import CaseResult,JudgeResult,Status


def implementation():
    name='posttrain.training.verl_pilot_reward'
    assert importlib.util.find_spec(name) is not None,'verl pilot reward bridge not implemented'
    return importlib.import_module(name)


def test_partial_reward_and_audit_bind_actual_frozen_denominator(tmp_path,problem):
    module=implementation()
    prompt=public_prompt(replace(problem,validation_status='reference_passed',cluster_id='cluster'),'pilot')
    class ScoreFixture:
        def score(self,problem_id,code,request_id):
            return JudgeResult(request_id,Status.WRONG,1,2,(CaseResult(Status.OK),CaseResult(Status.WRONG)),
                .6,.1,2,0,False,'fixture','isolated_execution',digest(code),problem.tests_hash)
    prompt={**prompt,'test_count':2}
    result=module.score_candidate('print(1)',prompt,ScoreFixture(),tmp_path,'a'*32,'mixed')
    assert result['score']==.25 and result['binary']==0
    assert result['queue_s']==.1 and result['judging_s']==.5
    record=json.loads((tmp_path/'records'/('a'*32+'.json')).read_text(encoding='utf-8'))
    assert record['state']=='complete' and record['result']['total']==2
    # The frozen extractor normalizes a trailing newline before submission.
    assert record['submitted_code']=='print(1)\n' and record['rewards']['mixed']==.25


def test_transport_timeout_aborts_without_numeric_zero_reward(tmp_path,problem):
    module=implementation()
    prompt=public_prompt(replace(problem,validation_status='reference_passed',cluster_id='cluster'),'pilot')
    class Unavailable:
        def score(self,*args):raise TimeoutError('test transport timeout')
        def reconcile(self,*args):raise OSError('test disconnected')
    with pytest.raises(module.PilotScoringFailure):
        module.score_candidate('print(1)',prompt,Unavailable(),tmp_path,'b'*32,'binary')
    record=json.loads((tmp_path/'records'/('b'*32+'.json')).read_text(encoding='utf-8'))
    assert record['state']=='unknown_or_invalid' and record['rewards'] is None
    assert not issubclass(module.PilotScoringFailure,TimeoutError)


def test_reference_identity_cannot_be_replaced_with_answer(problem):
    module=implementation()
    prompt=public_prompt(replace(problem,validation_status='reference_passed',cluster_id='cluster'),'pilot')
    identity={k:prompt[k] for k in ('problem_id','tests_hash','test_count')}
    module.check_input('posttrain_stdio',json.dumps(identity),prompt,prompt)
    for ground_truth in ('42',json.dumps({**identity,'test_count':1,'answer':'hidden'})):
        with pytest.raises(ValueError):module.check_input('posttrain_stdio',ground_truth,prompt,prompt)
    with pytest.raises(ValueError):
        module.check_input('final_humaneval',json.dumps(identity),prompt,prompt)


def test_unknown_request_cannot_be_silently_dispatched_again(tmp_path,problem):
    module=implementation()
    prompt=public_prompt(replace(problem,validation_status='reference_passed',cluster_id='cluster'),'pilot')
    class Unknown:
        def score(self,*args):raise TimeoutError('fixture')
        def reconcile(self,*args):raise OSError('fixture')
    with pytest.raises(module.PilotScoringFailure):
        module.score_candidate('print(1)',prompt,Unknown(),tmp_path,'c'*32,'binary')
    with pytest.raises(ValueError,match='existing_request_requires_reconciliation'):
        module.score_candidate('print(1)',prompt,Unknown(),tmp_path,'c'*32,'binary')
