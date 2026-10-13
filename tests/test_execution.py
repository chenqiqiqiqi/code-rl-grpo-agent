from dataclasses import replace

import pytest

from posttrain.data.prepare import validate_references
from posttrain.data.schema import TestCase as Case
from posttrain.execution.contracts import CaseResult, Limits, Status, cache_key
from posttrain.execution.judge import Judge
from posttrain.execution.replay import ReplayBackend
from posttrain.execution.sandboxfusion import parse_response
from posttrain.rewards import rewards, group_statistics


def judge_outcomes(outcomes,tests=None,**kwargs):
    backend=ReplayBackend(outcomes)
    judge=Judge(backend,backoff_s=0,**kwargs)
    result=judge.score('not executed',tests or tuple(Case(str(i),'yes') for i in range(len(outcomes))),Limits())
    return result,judge,backend


def test_complete_scoring_and_both_rewards():
    outcomes=[CaseResult(Status.OK,stdout='yes')]*8+[CaseResult(Status.OK,stdout='no')]*2
    result,_,backend=judge_outcomes(outcomes)
    assert len(backend.calls)==10 and result.valid
    assert rewards(result)=={'binary':0.0,'mixed':0.4}


@pytest.mark.parametrize('status',[Status.SYNTAX,Status.RUNTIME,Status.TIMEOUT,Status.MEMORY,Status.OUTPUT])
def test_program_errors_are_valid_and_do_not_short_circuit(status):
    result,_,backend=judge_outcomes([CaseResult(status),CaseResult(Status.OK,stdout='yes')])
    assert result.valid and len(backend.calls)==2
    assert rewards(result)=={'binary':0.0,'mixed':.25}


def test_service_failure_retries_exactly_two_and_is_invalid():
    failed=CaseResult(Status.INFRA,physical_executions=None,detail='queue_timeout')
    result,judge,backend=judge_outcomes([failed]*3,tests=(Case('a','yes'),))
    assert not result.valid and rewards(result) is None
    assert result.retries==2 and len(backend.calls)==3 and result.physical_executions is None
    assert judge.cache=={}


def test_retry_recovery_counts_physical_work():
    result,_,backend=judge_outcomes([CaseResult(Status.INFRA,physical_executions=0),CaseResult(Status.OK,stdout='yes')],tests=(Case('a','yes'),))
    assert result.valid and result.physical_executions==1 and result.retries==1


def test_cache_identity_and_no_duplicate_physical_execution():
    result,judge,backend=judge_outcomes([CaseResult(Status.OK,stdout='yes')],tests=(Case('a','yes'),))
    replay=judge.score('not executed',(Case('a','yes'),),Limits())
    assert replay.cache_hit and replay.physical_executions==0 and len(backend.calls)==1
    assert cache_key('a','t',Limits(),'v') != cache_key('a','t',Limits(memory_mb=512),'v')
    assert cache_key('a','t',Limits(),'v') != cache_key('a','t2',Limits(),'v')


def test_empty_and_conflicting_tests_never_call_backend():
    backend=ReplayBackend([])
    judge=Judge(backend)
    for tests in ((),(Case('a','1'),Case('a','2'))):
        result=judge.score('code',tests,Limits())
        assert result.status==Status.DATA and rewards(result) is None
    assert backend.calls==[]


def test_entire_invalid_group_removed_equal_groups_retained():
    good,_,_=judge_outcomes([CaseResult(Status.OK,stdout='yes')])
    bad=replace(good,status=Status.INFRA)
    stats=group_statistics([[good,bad],[good,good]],2)
    assert stats['valid_group_mask']==[False,True]
    assert stats['valid_groups']==1 and stats['binary']['signal_group_fraction']==0


def test_no_valid_group_is_not_zero_signal():
    result,_,_=judge_outcomes([CaseResult(Status.INFRA)]*3,tests=(Case('a','b'),))
    stats=group_statistics([[result,result]],2)
    assert stats['binary']['signal_group_fraction'] is None
    assert not stats['diagnostic_thresholds_met']


def test_mock_cannot_validate_reference(problem):
    with pytest.raises(ValueError,match='isolated_execution'):
        validate_references([problem],Judge(ReplayBackend([])))


@pytest.mark.parametrize('payload',[{}, {'status':'SandboxError'},{'status':'Success'}, {'status':'Success','run_result':{'status':'Mystery'}}, {'status':'Failed','run_result':{'status':'Finished','return_code':-9,'execution_time':.1}}])
def test_unknown_service_results_fail_closed(payload):
    assert parse_response(payload,Limits()).status==Status.INFRA


@pytest.mark.parametrize(('status','code','stderr','expected'),[
    ('Finished',0,'',Status.OK),('Finished',1,'SyntaxError',Status.SYNTAX),
    ('Finished',1,'MemoryError',Status.MEMORY),('TimeLimitExceeded',None,'',Status.TIMEOUT),
])
def test_service_known_results(status,code,stderr,expected):
    payload={'status':'Success' if code==0 else 'Failed','run_result':{'status':status,'return_code':code,'stdout':'','stderr':stderr,'execution_time':.1}}
    assert parse_response(payload,Limits()).status==expected


def test_healthy_local_supervisor_attributes_candidate_sigkill():
    payload={'status':'Failed','run_result':{'status':'Finished','return_code':-9,
             'stdout':'partial','stderr':'','execution_time':.3}}
    assert parse_response(payload,Limits(),supervised=True).status==Status.RUNTIME
    assert parse_response(payload,Limits()).status==Status.INFRA


def test_supervised_output_limit_preserves_bounded_output_and_timing():
    payload={'status':'Failed','project_event':'output_limit','run_result':{
        'status':'Finished','return_code':-9,'stdout':'x'*16,'stderr':'','execution_time':.3}}
    result=parse_response(payload,Limits(output_bytes=16),supervised=True)
    assert result.status==Status.OUTPUT and result.truncated
    assert result.stdout=='x'*16 and result.elapsed_s==.3 and result.exit_code==-9


@pytest.mark.parametrize('changes',[{'status':'Failed'}, {'stdout':False}, {'stderr':[]}, {'execution_time':True}])
def test_malformed_or_inconsistent_success_is_never_rewarded(changes):
    process={'status':'Finished','return_code':0,'stdout':'yes','stderr':'','execution_time':.3}
    payload={'status':'Success','run_result':process}
    for key,value in changes.items():
        if key=='status': payload[key]=value
        else: process[key]=value
    assert parse_response(payload,Limits()).status==Status.INFRA


def test_inconsistent_passed_count_is_not_a_valid_reward():
    result,_,_=judge_outcomes([CaseResult(Status.RUNTIME)],tests=(Case('a','yes'),))
    assert not replace(result,passed=1).valid
    assert rewards(replace(result,passed=1)) is None


def test_cache_replay_clears_nested_work_without_mutating_first_result():
    original, judge, backend = judge_outcomes(
        [CaseResult(Status.OK, stdout='yes', elapsed_s=.4, queue_s=.2)],
        tests=(Case('a','yes'),))
    replay = judge.score('not executed',(Case('a','yes'),),Limits())
    assert replay.cache_hit and len(backend.calls)==1
    assert replay.cases[0].physical_executions==0
    assert replay.cases[0].elapsed_s==0 and replay.cases[0].queue_s==0
    assert original.cases[0].physical_executions==1
    assert original.cases[0].elapsed_s==.4 and original.cases[0].queue_s==.2
    assert rewards(original)==rewards(replay)
