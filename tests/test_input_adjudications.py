from copy import deepcopy
import pytest
from posttrain.common import digest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.reviewed_contract_selection import select_reviewed_contract

STATEMENT = 'n is an even integer between 1 and 10.'
EVIDENCE = [{'start': 0, 'end': len(STATEMENT), 'text': STATEMENT}]


def fixture():
    return {'schema_version': 1, 'coverage': 'partial', 'layout_complete': True,
            'sound_for_deletion': True, 'limitations': ['general_parity_not_encoded'],
            'operations': [{'id': 'n', 'kind': 'fields', 'evidence': EVIDENCE,
                            'fields': [{'kind': 'int', 'name': 'n'}]}],
            'reviewed_invalid_inputs': [{'stdin_sha256': digest('3\n'),
                'reason': 'source_test_has_odd_n', 'evidence': EVIDENCE,
                'review_sha256': 'a' * 64}]}


def select(c):
    return select_reviewed_contract(STATEMENT, {digest(STATEMENT): {
        'review_basis': 'Exact source input separately checked against parity statement.', 'contract': c}})


def test_adjudication_only_matches_exact_input_and_never_proves_other_inputs():
    c = select(fixture())
    r = validate_input(c, '3\n')
    assert r['coverage'] == 'partial'
    assert r['violations'][0]['reason'] == 'reviewed_invalid_input'
    assert r['violations'][0]['review_sha256'] == 'a' * 64
    for text in ('2\n', '5\n', '3 \n'):
        r = validate_input(c, text)
        assert r['coverage'] == 'partial' and not r['violations']


@pytest.mark.parametrize('change', ['complete', 'hash', 'missing_evidence', 'wrong_evidence', 'duplicate'])
def test_malformed_or_overclaiming_adjudication_fails_closed(change):
    c = fixture()
    if change == 'complete':
        c['coverage'] = 'complete'; c['limitations'] = []
    elif change == 'hash': c['reviewed_invalid_inputs'][0]['stdin_sha256'] = 'not-a-hash'
    elif change == 'missing_evidence': c['reviewed_invalid_inputs'][0]['evidence'] = []
    elif change == 'wrong_evidence': c['reviewed_invalid_inputs'][0]['evidence'] = [{'start':0,'end':3,'text':'bad'}]
    elif change == 'duplicate': c['reviewed_invalid_inputs'].append(deepcopy(c['reviewed_invalid_inputs'][0]))
    with pytest.raises(ValueError): select(c)


def test_cleaner_removes_only_reviewed_case_and_keeps_domain_gate_closed(problem):
    from dataclasses import replace
    from posttrain.data.full_cleaning import clean_problem
    original = replace(problem, statement=STATEMENT)
    cleaned, audit = clean_problem(original, {}, lambda _: select(fixture()), validate_input,
                                   lambda _: {'status': 'no_confirmed_issue', 'issues': []}, minimum_tests=3)
    assert audit['removed_indices'] == [3]
    assert len(original.tests) == 5 and len(cleaned.tests) == 4
    assert audit['input_coverage'] == 'partial'
    assert audit['disposition'] == 'pending_domain_verification'
    assert audit['training_ready'] is False


def test_nested_adjudications_are_rejected_instead_of_silently_ignored():
    child = fixture()
    c = {'schema_version': 2, 'coverage': 'partial', 'layout_complete': True,
         'sound_for_deletion': True, 'count_var': 'n', 'evidence': EVIDENCE,
         'header': deepcopy(child), 'case': deepcopy(child), 'case_sums': []}
    with pytest.raises(ValueError): select(c)
    for name in ('header', 'case'):
        c[name].pop('reviewed_invalid_inputs')
    item = deepcopy(child['reviewed_invalid_inputs'][0])
    item['stdin_sha256'] = digest('1 3\n')
    c['reviewed_invalid_inputs'] = [item]
    r = validate_input(select(c), '1 3\n')
    assert r['coverage'] == 'partial'
    assert r['violations'][0]['reason'] == 'reviewed_invalid_input'
