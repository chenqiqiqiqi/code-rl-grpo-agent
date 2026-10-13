import pytest
from posttrain.training.rm_gate import control_gate

def rows(n=348):return [{'problem_id':str(i),'state':'complete','binary_success':i<84,'format_rejected':False} for i in range(n)]

def test_identical_controls():
    r=control_gate(rows(),rows());assert r['passed'] and r['flips']==0

def test_four_net_and_twenty_flips_allowed():
    a=rows();b=rows()
    for i in range(8):b[i]['binary_success']=False
    for i in range(84,96):b[i]['binary_success']=True
    r=control_gate(a,b);assert r['flips']==20 and r['net']==4 and r['passed']

def test_five_net_stops():
    a=rows();b=rows()
    for i in range(84,89):b[i]['binary_success']=True
    assert not control_gate(a,b)['passed']

def test_incomplete_is_not_model_failure():
    with pytest.raises(ValueError):control_gate(rows(),rows(347))
