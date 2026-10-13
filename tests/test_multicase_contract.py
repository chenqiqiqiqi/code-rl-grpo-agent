from copy import deepcopy
import pytest
from posttrain.common import digest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract


STATEMENT = 'Each input has t cases, 1 <= t <= 3; each case has n and n integers, 1 <= n <= 4 and 1 <= a_i <= n. Sum n <= 5.'
EVIDENCE = [{'start': 0, 'end': len(STATEMENT), 'text': STATEMENT}]


def expr(constant=0, **terms):
    return {'constant': constant, 'terms': terms}


def field(name, upper):
    return {'kind': 'int', 'name': name, 'evidence': EVIDENCE,
            'bounds': [{'min_expr': expr(1), 'max_expr': upper, 'evidence': EVIDENCE}]}


def flat(operations):
    return {'schema_version': 1, 'coverage': 'complete', 'layout_complete': True,
            'sound_for_deletion': True, 'operations': operations, 'semantics': [], 'constraints': []}


def contract():
    return {'schema_version': 2, 'coverage': 'complete', 'layout_complete': True,
            'sound_for_deletion': True, 'count_var': 't', 'evidence': EVIDENCE,
            'header': flat([{'id': 't', 'kind': 'fields', 'fields': [field('t', expr(3))]}]),
            'case': flat([{'id': 'n', 'kind': 'fields', 'fields': [field('n', expr(4))]},
                          {'id': 'a', 'kind': 'repeat', 'array': True, 'count_expr': expr(n=1),
                           'fields': [field('a_i', expr(n=1))]}]),
            'case_sums': [{'id': 'sum_n', 'expression': expr(n=1), 'limit': 5,
                           'operator': '<=', 'evidence': EVIDENCE}]}


def test_each_case_has_independent_length_bounds_and_global_token_offsets():
    c = contract()
    assert validate_input(c, '2 2 1 2 1 1') == {'coverage': 'complete', 'violations': []}
    r = validate_input(c, '2 2 1 2 1 2')
    assert r['violations'][0]['reason'] == 'integer_out_of_range'
    assert r['violations'][0]['token_index'] == 5
    assert r['violations'][0]['case_index'] == 1


def test_global_sum_is_checked_after_all_cases_and_extra_input_rejected():
    r = validate_input(contract(), '2 3 1 1 1 3 1 1 1')
    assert [v['reason'] for v in r['violations']] == ['case_sum_violated']
    assert r['violations'][0]['value'] == 6
    r = validate_input(contract(), '1 1 1 7')
    assert [v['reason'] for v in r['violations']] == ['trailing_input_tokens']


def test_missing_or_unparseable_cases_never_claim_complete_or_check_global_sum():
    for stdin in ['2 1 1', '2 x 1 1', '1000000000000', '1 -2']:
        r = validate_input(contract(), stdin)
        assert r['coverage'] == 'partial'
        assert not any(v['reason'] == 'case_sum_violated' for v in r['violations'])


def test_semantics_are_scoped_to_each_case():
    c = contract()
    c['case']['semantics'] = [{'id': 'unique', 'kind': 'all_distinct', 'operation_id': 'a',
                                'field_name': 'a_i', 'evidence': EVIDENCE}]
    assert not validate_input(c, '2 2 1 2 2 1 2')['violations']
    r = validate_input(c, '2 2 1 2 2 1 1')
    assert [v['case_index'] for v in r['violations']] == [1]


def test_release_gate_checks_nested_schema_and_nested_evidence():
    c = contract()
    reviewed = {digest(STATEMENT): {'review_basis': 'explicit two-level grammar', 'contract': c}}
    assert select_reviewed_contract(STATEMENT, reviewed)['schema_version'] == 2
    c['case']['operations'][0]['fields'][0]['evidence'] = [{'start': 0, 'end': 3, 'text': 'bad'}]
    with pytest.raises(ValueError, match='evidence_mismatch'):
        select_reviewed_contract(STATEMENT, reviewed)


@pytest.mark.parametrize('change', [
    lambda c: c.update(count_var='missing'),
    lambda c: c['case'].update(schema_version=2),
    lambda c: c['case'].update(coverage='partial'),
    lambda c: c['case_sums'][0].update(operator='>='),
    lambda c: c.update(operations=[]),
])
def test_invalid_multicase_schema_refused(change):
    c = deepcopy(contract()); change(c)
    with pytest.raises(ValueError, match='reviewed_contract_schema'):
        select_reviewed_contract(STATEMENT, {digest(STATEMENT): {'review_basis': 'test', 'contract': c}})
