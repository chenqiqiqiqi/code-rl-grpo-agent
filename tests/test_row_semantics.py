from copy import deepcopy
import pytest
from posttrain.common import digest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract


TEXT = 'n rows x y: x differs from y; pairs are distinct; gcd(x,y)=1; sum x<=k; x is a multiple of 100.'
EV = [{'start': 0, 'end': len(TEXT), 'text': TEXT}]


def expression(constant=0, **terms):
    return {'constant': constant, 'terms': terms}


def fixture(kind, **extra):
    return {'schema_version': 1, 'coverage': 'complete', 'sound_for_deletion': True,
            'operations': [
                {'id': 'header', 'kind': 'fields', 'fields': [{'kind': 'int', 'name': 'n'}, {'kind': 'int', 'name': 'k'}]},
                {'id': 'rows', 'kind': 'repeat', 'rows': True, 'count_expr': expression(n=1),
                 'fields': [{'kind': 'int', 'name': 'x'}, {'kind': 'int', 'name': 'y'}]}],
            'semantics': [{'id': 'check', 'kind': kind, 'operation_id': 'rows', 'evidence': EV, **extra}]}


@pytest.mark.parametrize('kind,extra,valid,invalid', [
    ('row_pairwise_neq', {'field_names': ['x', 'y']}, '1 0 1 2', '1 0 1 1'),
    ('unique_tuple_rows', {'field_names': ['x', 'y']}, '2 0 1 1 1 2', '2 0 1 1 1 1'),
    ('row_gcd_one', {'field_names': ['x', 'y']}, '1 0 2 3', '1 0 6 9'),
    ('array_sum_upper_bound', {'field_name': 'x', 'expression': expression(k=1)}, '2 5 2 1 3 1', '2 4 2 1 3 1'),
    ('array_sum_lower_bound', {'field_name': 'x', 'expression': expression(k=1)}, '2 5 2 1 3 1', '2 6 2 1 3 1'),
    ('array_sum_equals', {'field_name': 'x', 'expression': expression(k=1)}, '2 5 2 1 3 1', '2 4 2 1 3 1'),
    ('aggregate_gcd_one', {'field_name': 'x'}, '2 0 2 1 3 1', '2 0 6 1 9 1'),
    ('row_sum_lower_bound', {'field_names': ['x','y'], 'expression': expression(1)}, '1 0 0 1', '1 0 0 0'),
    ('row_sum_upper_bound', {'field_names': ['x','y'], 'expression': expression(k=1)}, '1 5 2 3', '1 4 2 3'),
    ('unique_unordered_pair_rows', {'field_names': ['x','y']}, '2 0 1 2 1 3', '2 0 1 2 2 1'),
    ('multiple_of_constant', {'field_name': 'x', 'divisor': 100}, '2 0 0 1 200 1', '1 0 101 1'),
])
def test_reviewed_row_rules_use_requested_field_tuple(kind, extra, valid, invalid):
    c = fixture(kind, **extra)
    select_reviewed_contract(TEXT, {digest(TEXT): {'review_basis': 'explicit rule', 'contract': c}})
    assert not validate_input(c, valid)['violations']
    failures = validate_input(c, invalid)['violations']
    assert len(failures) == 1
    assert failures[0]['reason'] == kind + '_violated'
    assert failures[0]['evidence'] == EV


def test_unknown_or_incomplete_layout_never_invents_tuple_violations():
    c = fixture('unique_tuple_rows', field_names=['x', 'y'])
    r = validate_input(c, '2 0 1 1 1')
    assert r['coverage'] == 'partial'
    assert not any(v['reason'] == 'unique_tuple_rows_violated' for v in r['violations'])


def test_scalar_tuple_and_scalar_multiple_use_values_from_that_operation():
    c = fixture('row_pairwise_neq', field_names=['n', 'k'])
    c['semantics'][0]['operation_id'] = 'header'
    select_reviewed_contract(TEXT, {digest(TEXT): {'review_basis': 'test', 'contract': c}})
    assert not validate_input(c, '1 2 0 0')['violations']
    assert validate_input(c, '1 1 0 0')['violations'][0]['reason'] == 'row_pairwise_neq_violated'
    c = fixture('multiple_of_constant', field_name='k', divisor=100)
    c['semantics'][0]['operation_id'] = 'header'
    select_reviewed_contract(TEXT, {digest(TEXT): {'review_basis': 'test', 'contract': c}})
    assert not validate_input(c, '1 200 0 0')['violations']
    assert validate_input(c, '1 201 0 0')['violations'][0]['reason'] == 'multiple_of_constant_violated'


def test_point_tuple_inequality_allows_equal_individual_coordinates():
    c=fixture('row_tuple_neq',field_groups=[['x','y'],['u','v']])
    c['operations'][1]['fields'].extend([{'kind':'int','name':'u'},{'kind':'int','name':'v'}])
    select_reviewed_contract(TEXT,{digest(TEXT):{'review_basis':'points','contract':c}})
    assert not validate_input(c,'1 0 1 1 1 2')['violations']
    assert validate_input(c,'1 0 1 1 1 1')['violations'][0]['reason']=='row_tuple_neq_violated'


def test_row_linear_expression_uses_current_row_and_scalar_context():
    c=fixture('row_linear_upper_bound',expression=expression(x=2,k=-1),limit=0)
    select_reviewed_contract(TEXT,{digest(TEXT):{'review_basis':'linear','contract':c}})
    assert not validate_input(c,'2 6 2 0 3 0')['violations']
    assert validate_input(c,'2 6 2 0 4 0')['violations'][0]['reason']=='row_linear_upper_bound_violated'


@pytest.mark.parametrize('rule', [
    {'kind': 'row_pairwise_neq', 'field_names': ['x']},
    {'kind': 'row_pairwise_neq', 'field_names': ['x', 'missing']},
    {'kind': 'row_pairwise_neq', 'field_names': ['x', 'x']},
    {'kind': 'multiple_of_constant', 'field_name': 'x', 'divisor': 0},
    {'kind': 'multiple_of_constant', 'field_name': 'x', 'divisor': True},
])
def test_invalid_semantic_parameters_fail_closed(rule):
    c = deepcopy(fixture(**rule))
    with pytest.raises(ValueError, match='reviewed_contract_schema'):
        select_reviewed_contract(TEXT, {digest(TEXT): {'review_basis': 'test', 'contract': c}})
