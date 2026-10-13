from dataclasses import replace
import pytest
from posttrain.execution.contracts import CaseResult,JudgeResult,Status
from posttrain.execution.service import checked_reply
from posttrain.common import digest


def reply():
    return JudgeResult('req',Status.OK,1,1,(CaseResult(Status.OK,stdout='yes'),),.1,0,1,0,False,
                       'executor-v1','isolated_execution',digest('code'),'tests').to_dict()


def test_remote_score_binds_request_code_tests_and_executor():
    args=('req','code','tests','executor-v1',1)
    assert checked_reply(reply(),*args).valid
    for key in ('request_id','code_hash','tests_hash','executor_version'):
        with pytest.raises(ValueError):checked_reply({**reply(),key:'wrong'},*args)


def test_remote_unknown_or_inconsistent_counts_fail_closed():
    for key,value in [('passed',True),('passed',0),('status','unknown'),('valid',False)]:
        with pytest.raises(ValueError):checked_reply({**reply(),key:value},'req','code','tests','executor-v1',1)


def test_remote_cannot_shrink_frozen_test_denominator():
    with pytest.raises(ValueError,match='frozen_test_count_mismatch'):
        checked_reply(reply(),'req','code','tests','executor-v1',2)


def test_pending_request_survives_restart_and_cannot_be_executed_twice(tmp_path):
    from posttrain.execution.service import RequestJournal
    identity={'executor_version':'v1','problem_set_sha256':'dataset'}
    request={'request_id':'crashed','code_hash':digest('code'),'tests_hash':'tests','problem_id':'p'}
    first=RequestJournal(tmp_path/'requests.sqlite',identity)
    assert first.claim(request)
    restarted=RequestJournal(tmp_path/'requests.sqlite',identity)
    assert not restarted.claim(request)
    receipt=restarted.lookup('crashed')
    assert receipt['state']=='unknown_after_restart'
    assert receipt['request']==request and receipt['result'] is None
    assert restarted.lookup('missing') is None


def test_completed_request_can_be_reconciled_but_never_overwritten(tmp_path):
    from posttrain.execution.service import RequestJournal
    journal=RequestJournal(tmp_path/'requests.sqlite',{'executor_version':'v1'})
    request={'request_id':'req','code_hash':digest('code')}
    assert journal.claim(request)
    journal.complete('req',reply())
    restarted=RequestJournal(tmp_path/'requests.sqlite',{'executor_version':'v1'})
    assert digest(restarted.lookup('req')['result'])==digest(reply())
    assert not restarted.claim(request)
    with pytest.raises(ValueError,match='already_completed'):
        restarted.complete('req',{**reply(),'passed':0})
    with pytest.raises(ValueError,match='gateway_identity_changed'):
        RequestJournal(tmp_path/'requests.sqlite',{'executor_version':'v2'})


def test_legacy_result_reconciles_over_http_with_full_identity_checks(tmp_path,problem):
    import threading
    from posttrain.common import write_json
    from posttrain.execution.service import make_server,ScoringClient
    class NeverDispatch:
        version='executor-v1'
        evidence_level='isolated_execution'
        def run(self,*args):raise AssertionError('legacy_reconciliation_must_not_execute')
    p=replace(problem,validation_status='reference_passed')
    row=JudgeResult('req',Status.OK,len(p.tests),len(p.tests),tuple(CaseResult(Status.OK) for _ in p.tests),
        .1,0,len(p.tests),0,False,'executor-v1','isolated_execution',digest('code'),p.tests_hash).to_dict()
    write_json(tmp_path/'req.json',row)
    server=make_server([p],NeverDispatch(),tmp_path,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=ScoringClient(f'http://127.0.0.1:{server.server_port}',NeverDispatch.version,
            {p.problem_id:{'tests_hash':p.tests_hash,'test_count':len(p.tests)}})
        assert client.reconcile(p.problem_id,'code','req').valid
        with pytest.raises(ValueError):client.reconcile(p.problem_id,'different','req')
    finally:server.shutdown();server.server_close();thread.join()
