import pytest
from posttrain.training.selfdistill import admission_reason
from posttrain.common import digest

def fixture():
    c={'request_id':'r','problem_id':'p','text':'print(1)','output_tokens':4,'finish_reason':'stop'}
    r={'request_id':'r','problem_id':'p','state':'complete','binary_success':True,'format_rejected':False,
       'code_hash':digest('print(1)\n'),'output_tokens':4,
       'result':{'binary_valid':True,'evidence_level':'isolated_execution','tests_hash':'h','total':2,'passed':2,
                 'cases':[{'status':'ok'},{'status':'ok'}]}}
    p={'problem_id':'p','tests_hash':'h','test_count':2}
    return c,r,p

def test_only_full_isolated_success_is_admitted():
    c,r,p=fixture();assert admission_reason(c,r,p) is None
    r['result']['cases'].pop()
    with pytest.raises(ValueError):admission_reason(c,r,p)

def test_changed_code_and_unknown_state_fail_closed():
    c,r,p=fixture();c['text']='print(2)'
    with pytest.raises(ValueError):admission_reason(c,r,p)
    c,r,p=fixture();r['state']='unknown_or_invalid'
    with pytest.raises(ValueError):admission_reason(c,r,p)

def test_capped_output_is_not_a_training_target():
    c,r,p=fixture();c['finish_reason']='length'
    assert admission_reason(c,r,p)=='not_natural_stop'
