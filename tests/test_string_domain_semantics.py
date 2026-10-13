import pytest
from posttrain.common import digest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract

S = 's has lowercase letters, t is a subsequence of s; positive lengths and stated character constraints.'
EV = [{'start':0,'end':len(S),'text':S}]


def expr(constant=0, **terms):
    return {'constant':constant,'terms':terms}


def string(name, **changes):
    return {'kind':'string','name':name,'alphabet':'abc','min_length_expr':expr(1),
            'max_length_expr':expr(10),'evidence':EV, **changes}


def contract(fields):
    return {'schema_version':1,'coverage':'complete','sound_for_deletion':True,
            'operations':[{'id':'input','kind':'fields','fields':fields}]}


@pytest.mark.parametrize('options,valid,invalid,reason',[
    ({'contains_all':'ab'}, 'abc', 'aac', 'string_required_character_missing'),
    ({'is_palindrome':True}, 'aba', 'abc', 'string_not_palindrome'),
    ({'starts_with':'a','ends_with':'b'}, 'acb', 'bca', 'string_prefix_mismatch'),
    ({'distinct_characters':True}, 'abc', 'aba', 'string_duplicate_character'),
    ({'character_counts':{'a':{'min':1,'max':1}}}, 'abc', 'aab', 'string_character_count_out_of_range'),
    ({'character_count_moduli':{'a':2}}, 'aab', 'ab', 'string_character_count_modulus_violation'),
    ({'forbidden_prefixes':['a']}, 'bc', 'ac', 'string_forbidden_prefix'),
])
def test_declared_string_properties(options,valid,invalid,reason):
    c=contract([string('s',**options)])
    select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
    assert not validate_input(c,valid)['violations']
    assert reason in [v['reason'] for v in validate_input(c,invalid)['violations']]


def test_length_binding_is_explicit_and_available_to_later_fields():
    c=contract([string('s',length_binding='s_length'), string('t',length_expr=expr(s_length=1))])
    select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
    assert validate_input(c,'ab cc')=={'coverage':'complete','violations':[]}
    assert validate_input(c,'ab c')['violations'][0]['reason']=='string_length_mismatch'


def test_subsequence_relationship_does_not_mean_substring():
    c=contract([string('s'),string('t',subsequence_of='s')])
    select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
    assert not validate_input(c,'abc ac')['violations']
    assert validate_input(c,'abc ca')['violations'][0]['reason']=='string_not_subsequence'
    c=contract([string('t',subsequence_of='unknown')])
    assert validate_input(c,'a')['coverage']=='partial'


def test_dynamic_alphabet_uses_declared_order_and_count():
    c=contract([{'kind':'int','name':'m'},string('s',alphabet_prefix_length_expr=expr(m=1))])
    select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
    assert not validate_input(c,'2 ba')['violations']
    assert validate_input(c,'2 ca')['violations'][0]['reason']=='string_alphabet_prefix_violation'


def test_product_limit_binds_string_lengths_without_evaluating_code():
    c=contract([string('s',length_binding='ls'),string('t',length_binding='lt')])
    c['constraints']=[{'id':'product','kind':'product_upper_bound','factors':[expr(ls=1),expr(lt=1)],'limit':5,'evidence':EV}]
    select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
    assert not validate_input(c,'ab ab')['violations']
    assert validate_input(c,'ab abc')['violations'][0]['reason']=='constraint_violated'


@pytest.mark.parametrize('options',[
    {'contains_all':'x'}, {'distinct_characters':'true'}, {'length_binding':''},
    {'starts_with':' '}, {'is_palindrome':1}, {'subsequence_of':5},
    {'character_counts':{'a':{'min':2,'max':1}}}, {'character_count_moduli':{'a':0}},
    {'forbidden_prefixes':['x']},
])
def test_unknown_or_unsafe_string_rule_rejected(options):
    c=contract([string('s',**options)])
    with pytest.raises(ValueError,match='reviewed_contract_schema'):
        select_reviewed_contract(S,{digest(S):{'review_basis':'explicit','contract':c}})
