import copy
import pytest
from posttrain.training.sft_dual_contract import route_plan,validate_routes,route_weights

def sample(pid,role,cluster=None):
    return {'problem_id':pid,'cluster_id':cluster or pid,'role':role,'sample':{'input_ids':[1,2,3,4], 'attention_mask':[1,1,1,1], 'labels':[-100,-100,3,4]},'admission':'semantic_and_formal_pass' if role=='supervised_gold' else 'frozen_m0_trajectory'}

def test_reject_failed_trajectory_as_ce_gold():
    with pytest.raises(ValueError,match='supervision'):validate_routes([sample('s','preservation_only')],[sample('a','preservation_only')],[])

def test_reject_shared_cluster():
    with pytest.raises(ValueError,match='cluster'):validate_routes([sample('s','supervised_gold','same')],[sample('a','preservation_only','same')],[])

def test_reject_holdout_leak():
    with pytest.raises(ValueError,match='cluster'):validate_routes([sample('s','supervised_gold')],[sample('a','preservation_only')],['s'])

def test_independent_denominators():
    assert route_weights([2,6],[3,9,12])==([.25,.75],[.125,.375,.5])

def test_all_preservation_rows_every_update_and_resume_suffix():
    s,a=route_plan(32,16,8,13,42)
    assert len(s)==len(a)==13 and all(len(x)==8 for x in s)
    assert all(sorted(x)==list(range(16)) for x in a)
    assert route_plan(32,16,8,13,42)==(s,a)
    assert s[7:]==route_plan(32,16,8,13,42)[0][7:]

def test_invalid_target_rejected():
    x=sample('s','supervised_gold');x['sample']['labels']=[-100]*4
    with pytest.raises(ValueError):validate_routes([x],[sample('a','preservation_only')],[])
