import pytest
from posttrain.common import digest
from posttrain.data.reviewed_contract_selection import select_reviewed_contract


def entry_for(s):
    return {'review_basis':'explicit scalar','contract':{'schema_version':1,'coverage':'complete',
        'sound_for_deletion':True,'operations':[{'id':'n','kind':'fields',
        'evidence':[{'start':0,'end':len(s),'text':s}], 'fields':[{'kind':'int','name':'n'}]}]}}


def test_unreviewed_statement_never_inherits_heuristic_trust():
    c=select_reviewed_contract('Input: n',{})
    assert c['coverage']=='unsupported' and not c['sound_for_deletion']
    assert c['operations']==[]


def test_reviewed_contract_is_bound_to_exact_statement_and_not_mutated():
    s='integer n';entry=entry_for(s)
    c=select_reviewed_contract(s,{digest(s):entry})
    assert 'release_review_sha256' in c
    assert 'release_review_sha256' not in entry['contract']
    assert select_reviewed_contract(s+' changed',{digest(s):entry})['coverage']=='unsupported'


def test_changed_evidence_is_rejected():
    s='integer n';entry=entry_for(s)
    entry['contract']['operations'][0]['evidence'][0]['text']='integer x'
    with pytest.raises(ValueError,match='evidence_mismatch'):
        select_reviewed_contract(s,{digest(s):entry})


@pytest.mark.parametrize('change',[
    lambda c:c.update(schema_version=2),
    lambda c:c['operations'][0].update(kind='opaque_suffix'),
    lambda c:c['operations'][0].update(fields=[]),
    lambda c:c['operations'][0]['fields'][0].update(kind='decimal'),
    lambda c:c['operations'][0].update(distinct=True),
    lambda c:c.update(semantics=[{'kind':'sorted','operation_id':'n'}]),
    lambda c:c.update(constraints=[{'kind':'prime'}]),
    lambda c:c['operations'][0]['fields'][0].update(alphabet=123),
    lambda c:c['operations'][0]['fields'][0].update(bounds=[{'min_expr':{'constant':'1','terms':{}}}]),
    lambda c:c.update(limitations=['some_semantic_constraints_not_compiled']),
])
def test_unimplemented_contract_checks_cannot_authorize_deletion(change):
    s='integer n';entry=entry_for(s);change(entry['contract'])
    with pytest.raises(ValueError,match='reviewed_contract_schema'):
        select_reviewed_contract(s,{digest(s):entry})


def test_output_hold_evidence_is_checked_before_freezing():
    s='integer n';entry=entry_for(s)
    entry['contract']['output_review_holds']=[{'reason':'manual_non_unique_output','evidence':[{'start':0,'end':len(s),'text':'changed'}]}]
    with pytest.raises(ValueError,match='evidence_mismatch'):
        select_reviewed_contract(s,{digest(s):entry})
    entry['contract']['output_review_holds'][0]['evidence'][0]['text']=s
    assert select_reviewed_contract(s,{digest(s):entry})['output_review_holds']


def test_complete_contract_cannot_claim_incomplete_layout():
    s='integer n'
    entry=entry_for(s)
    entry['contract']['layout_complete']=False
    with pytest.raises(ValueError,match='reviewed_contract_schema'):
        select_reviewed_contract(s,{digest(s):entry})


@pytest.mark.parametrize('field',[
    {'kind':'string','name':'s','length_expr':{'constant':3,'terms':{}}},
    {'kind':'string','name':'s','alphabet':'abc','length_expr':{'constant':0,'terms':{}}},
    {'kind':'string','name':'s','alphabet':'abc'},
    {'kind':'string','name':'s','alphabet':'a ','length_expr':{'constant':2,'terms':{}}},
])
def test_complete_contract_rejects_string_domains_that_tokens_cannot_represent(field):
    s='one non-empty token string'
    entry={'review_basis':'explicit string domain','contract':{
        'schema_version':1,'coverage':'complete','sound_for_deletion':True,'limitations':[],
        'operations':[{'id':'s','kind':'fields','evidence':[{'start':0,'end':len(s),'text':s}],
                       'fields':[field | {'evidence':[{'start':0,'end':len(s),'text':s}]}]}],
        'constraints':[],'semantics':[]}}
    with pytest.raises(ValueError,match='reviewed_contract_schema'):
        select_reviewed_contract(s,{digest(s):entry})


def test_complete_dynamic_string_length_is_safe_when_prior_bound_proves_positive():
    s='N is between 1 and 4; S has length N and uses only a or b.'
    evidence=[{'start':0,'end':len(s),'text':s}]
    entry={'review_basis':'explicit positive length and alphabet','contract':{
        'schema_version':1,'coverage':'complete','sound_for_deletion':True,'limitations':[],
        'operations':[
            {'id':'head','kind':'fields','evidence':evidence,'fields':[
                {'kind':'int','name':'N','evidence':evidence,'bounds':[{
                    'min_expr':{'constant':1,'terms':{}},
                    'max_expr':{'constant':4,'terms':{}},'evidence':evidence}]}]},
            {'id':'text','kind':'fields','evidence':evidence,'fields':[
                {'kind':'string','name':'S','alphabet':'ab','evidence':evidence,
                 'length_expr':{'constant':0,'terms':{'N':1}}}]}],
        'constraints':[],'semantics':[]}}
    assert select_reviewed_contract(s,{digest(s):entry})['coverage']=='complete'
