from posttrain.data.constraint_engine import validate_input


def contract(header_r=False):
    ev=[{'start':0,'end':1,'text':'x'}]
    def field(name,lo=None,hi=None):
        return {'kind':'int','name':name,'evidence':ev,'bounds':[{'min_expr':lo,'max_expr':hi,'evidence':ev}]}
    const=lambda n:{'constant':n,'terms':{}}
    var=lambda n:{'constant':0,'terms':{n:1}}
    return {'schema_version':1,'coverage':'complete','operations':[
        {'id':'head','kind':'fields','evidence':ev,'fields':[field('t',const(1),const(10))]+([field('r')] if header_r else [])},
        {'id':'queries','kind':'repeat','rows':True,'count_expr':var('t'),'evidence':ev,
         'fields':[field('l',const(0),var('r')),field('r',const(0),const(100))]}], 'semantics':[],'constraints':[]}


def test_row_relations_resolve_forward_fields_in_the_same_row():
    out=validate_input(contract(),'2 5 6 1 2')
    assert out['violations']==[]
    assert out['coverage']=='complete'


def test_later_row_does_not_hide_earlier_invalid_relation():
    out=validate_input(contract(),'2 7 6 1 10')
    assert any(v['reason']=='integer_out_of_range' and v['token_index']==1 for v in out['violations'])


def test_row_field_shadows_same_named_header_scalar():
    out=validate_input(contract(True),'2 0 5 6 1 2')
    assert out['violations']==[] and out['coverage']=='complete'


def test_missing_row_field_is_unknown_not_invented_relation_failure():
    out=validate_input(contract(),'1 5')
    assert out['coverage']=='partial'
    assert not any(v['reason']=='integer_out_of_range' for v in out['violations'])
