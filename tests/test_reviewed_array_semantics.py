from copy import deepcopy
import pytest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract
from posttrain.common import digest

S='n integers are strictly increasing and different from X.'
E=[{'start':0,'end':len(S),'text':S}]

def contract(kind,**kwargs):
    f=lambda name:{'kind':'int','name':name,'evidence':E,'bounds':[]}
    return {'schema_version':1,'coverage':'complete','sound_for_deletion':True,'limitations':[],
      'operations':[{'id':'head','kind':'fields','fields':[f('n'),f('X')],'evidence':E},
      {'id':'values','kind':'repeat','array':True,'count_expr':{'constant':0,'terms':{'n':1}},'fields':[f('a')],'evidence':E}],
      'constraints':[],'semantics':[{'id':'rule','kind':kind,'operation_id':'values','field_name':'a','evidence':E,**kwargs}]}

@pytest.mark.parametrize('kind,params,good,bad',[
 ('all_distinct',{},'3 9 1 2 3','3 9 1 2 1'),
 ('ordered',{'operator':'lt'},'3 9 1 2 3','3 9 1 2 2'),
 ('ordered',{'operator':'le'},'3 9 1 1 2','3 9 1 3 2'),
 ('excludes_value',{'expression':{'constant':0,'terms':{'X':1}}},'3 9 1 2 3','3 9 1 9 3'),
])
def test_array_rules_check_the_declared_operation(kind,params,good,bad):
    c=contract(kind,**params)
    select_reviewed_contract(S,{digest(S):{'review_basis':'test explicit relation','contract':c}})
    assert validate_input(c,good)['violations']==[]
    assert validate_input(c,bad)['violations']


def test_array_rule_does_not_mix_repeated_field_names_in_other_operations():
    c=contract('all_distinct')
    other=deepcopy(c['operations'][1]);other['id']='other';c['operations'].append(other)
    assert validate_input(c,'2 9 1 2 1 2')['violations']==[]


def test_permutation_uses_only_its_declared_operation():
    c=contract('permutation',length_var='n')
    other=deepcopy(c['operations'][1]);other['id']='other';c['operations'].append(other)
    assert validate_input(c,'2 9 1 2 8 8')['violations']==[]
    assert any(v['reason']=='not_a_permutation' for v in validate_input(c,'2 9 1 1 1 2')['violations'])


def test_unresolved_array_length_does_not_invent_a_domain_violation():
    c=contract('all_distinct');c['operations'][1]['count_expr']['terms']={'missing':1}
    out=validate_input(c,'2 9 1 1');assert out['coverage']=='partial' and not out['violations']


def test_unknown_order_operator_is_rejected_before_use():
    c=contract('ordered',operator='approximately')
    with pytest.raises(ValueError):select_reviewed_contract(S,{digest(S):{'review_basis':'test','contract':c}})
