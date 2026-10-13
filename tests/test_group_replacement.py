from dataclasses import replace

import pytest

from posttrain.execution.contracts import CaseResult,JudgeResult,Status
from posttrain.training.budget import TokenLedger,BudgetExceeded
from posttrain.training.groups import CandidateGroup,gather_valid_batch,IncompleteBatch
from posttrain.training.costs import CostLedger


def scored(ok=True):
    return JudgeResult('r',Status.OK if ok else Status.INFRA,1,1,(CaseResult(Status.OK),),0,0,1,0,False,'mock','mock_contract','c','t')


def test_whole_group_replacement_keeps_equal_rewards(tmp_path):
    ledger=TokenLedger(tmp_path/'tokens.db',1000)
    calls=[]
    def generate(ids):
        calls.extend(ids)
        return [CandidateGroup(x,(3,3),(scored(),scored(x!='invalid'))) for x in ids]
    batch=gather_valid_batch(['valid','invalid'],['replacement','unused'],2,20,ledger,generate,'binary')
    assert calls==['valid','invalid','replacement']
    assert [g.problem_id for g in batch['groups']]==['valid','replacement']
    assert batch['rewards']==[[1.,1.],[1.,1.]]
    assert ledger.used()==18
    ledger.close()


def test_no_partial_update_or_unbounded_resampling(tmp_path):
    ledger=TokenLedger(tmp_path/'tokens.db',1000)
    calls=[]
    def generate(ids):
        calls.extend(ids)
        return [CandidateGroup(x,(2,2),(scored(False),scored())) for x in ids]
    with pytest.raises(IncompleteBatch):
        gather_valid_batch(['p'],['r','extra'],2,10,ledger,generate,'mixed')
    assert calls==['p','r'] and ledger.used()==8
    ledger.close()


def test_replacement_needs_worst_case_budget(tmp_path):
    ledger=TokenLedger(tmp_path/'tokens.db',40)
    def generate(ids):
        return [CandidateGroup(x,(20,20),(scored(False),scored())) for x in ids]
    with pytest.raises(BudgetExceeded):
        gather_valid_batch(['p'],['r'],2,20,ledger,generate,'mixed')
    assert ledger.used()==40
    ledger.close()


def test_cost_idempotency_shared_cost_once_and_emergency_not_preallocated(tmp_path):
    ledger=CostLedger(tmp_path/'costs.db')
    ledger.add('pilot-1','shared','gpu','10.25','invoice:1')
    ledger.add('pilot-1','shared','gpu','10.25','invoice:1')
    assert str(ledger.total())=='10.25'
    assert ledger.can_start('10','1000')
    assert not ledger.can_start('100','1500')
    with pytest.raises(ValueError):
        ledger.add('pilot-1','shared','gpu','0','invoice:1')
    ledger.close()
