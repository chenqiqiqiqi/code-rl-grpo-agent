from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract
from posttrain.common import digest

S='S has length 1 through n and contains only a, b, c.'
E=[{'start':0,'end':len(S),'text':S}]

def contract():
 return {'schema_version':1,'coverage':'complete','sound_for_deletion':True,'limitations':[],
 'operations':[{'id':'head','kind':'fields','evidence':E,'fields':[{'kind':'int','name':'n','evidence':E}]},
 {'id':'string','kind':'fields','evidence':E,'fields':[{'kind':'string','name':'S','alphabet':'abc','evidence':E,
 'min_length_expr':{'constant':1,'terms':{}},'max_length_expr':{'constant':0,'terms':{'n':1}}}]}],
 'semantics':[],'constraints':[]}

def test_alphabet_and_length_rules_are_enforced():
 c=contract();select_reviewed_contract(S,{digest(S):{'review_basis':'test explicit domain','contract':c}})
 assert validate_input(c,'3 abc')['violations']==[]
 assert any(v['reason']=='string_alphabet_violation' for v in validate_input(c,'3 abD')['violations'])
 assert any(v['reason']=='string_length_out_of_range' for v in validate_input(c,'2 abc')['violations'])

def test_unknown_length_bound_stays_partial():
 c=contract();c['operations'][1]['fields'][0]['max_length_expr']['terms']={'missing':1}
 o=validate_input(c,'3 abc');assert o['coverage']=='partial' and not o['violations']


def test_row_local_string_length_shadows_header_length():
 c=contract();c['operations'][0]['fields'][0]['name']='n'
 f=c['operations'][1]['fields'][0];f.pop('min_length_expr');f.pop('max_length_expr');f['length_expr']={'constant':0,'terms':{'n':1}}
 c['operations'][1].update(kind='repeat',rows=True,count_expr={'constant':1,'terms':{}},fields=[{'kind':'int','name':'n','evidence':E},f])
 o=validate_input(c,'99 2 ab');assert not o['violations'] and o['coverage']=='complete'


def test_partial_opaque_string_preserves_numeric_prefix_and_blocks_suffix():
 c={'coverage':'partial','layout_complete':True,'operations':[
  {'id':'prefix','kind':'fields','fields':[{'kind':'int','name':'x','bounds':[
   {'min_expr':{'constant':1,'terms':{}},'max_expr':{'constant':3,'terms':{}},'evidence':[]}]}]},
  {'id':'text','kind':'fields','fields':[
   {'kind':'string','name':'s'},
   {'kind':'int','name':'y','bounds':[
    {'min_expr':{'constant':1,'terms':{}},'max_expr':{'constant':3,'terms':{}},'evidence':[]}]}]}]}
 o=validate_input(c,'0 alpha beta 4')
 assert o['coverage']=='partial'
 assert [v['rule_id'] for v in o['violations']]==['x:range0']


def test_partial_alphabet_only_string_does_not_authorize_token_deletion():
 c={'coverage':'partial','layout_complete':True,'operations':[
  {'id':'s','kind':'fields','fields':[{'kind':'string','name':'s','alphabet':'.#'}]}]}
 o=validate_input(c,'x')
 assert o=={'coverage':'partial','violations':[]}


def test_contradictory_complete_layout_is_runtime_partial_without_layout_checks():
 c={'coverage':'complete','layout_complete':False,'operations':[
  {'id':'n','kind':'fields','fields':[{'kind':'int','name':'n'}]}]}
 o=validate_input(c,'1 trailing')
 assert o=={'coverage':'partial','violations':[]}
