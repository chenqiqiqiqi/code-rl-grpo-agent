from dataclasses import replace

import pytest

from posttrain.common import digest
from posttrain.data.schema import Problem, TestCase as Case
from posttrain.data.requalification import join_verified_fresh_results


def evidence():
    p = Problem('p', 'synthetic', 'v1', 'p', 'Print the input.',
                tuple(Case(str(i), str(i)) for i in range(5)), ('print(input())',),
                cluster_id='cluster-p', validation_status='cleaned_pending_validation')
    audit = {'problem_id': 'p', 'statement_sha256': digest(p.statement),
             'cleaned_tests_hash': p.tests_hash, 'retained_test_count': 5,
             'input_coverage': 'complete', 'public_samples_checked': 1,
             'statement_assessment': {'status': 'no_confirmed_issue'},
             'checker_review_flags': [], 'oracle_review_required': False,
             'reliable_candidate_pending_fresh_backtest': False,
             'historical_reference_evidence': {'all_retained_tests_passed': False}}
    fresh = replace(p, validation_status='reference_passed').to_dict()
    return p.to_dict(), fresh, audit


def test_fresh_pass_can_rescue_historical_failure():
    source, fresh, audit = evidence()
    result = join_verified_fresh_results([source], [fresh], [audit])
    assert result['eligible_ids'] == ['p']
    assert result['training_ready'] is False


def test_old_pass_does_not_replace_missing_fresh_pass():
    source, _, audit = evidence()
    audit['reliable_candidate_pending_fresh_backtest'] = True
    result = join_verified_fresh_results([source], [], [audit])
    assert not result['eligible_ids']
    assert 'no_fresh_reference_pass' in result['pending'][0]['reasons']


def test_fresh_answer_or_reference_mutation_is_rejected():
    source, fresh, audit = evidence()
    fresh['tests'] = [{'stdin': str(i), 'stdout': 'changed'} for i in range(5)]
    with pytest.raises(ValueError, match='accepted_problem_changed'):
        join_verified_fresh_results([source], [fresh], [audit])
    source, fresh, audit = evidence()
    fresh['references'] = ['unregistered code']
    with pytest.raises(ValueError, match='accepted_reference_not_from_source'):
        join_verified_fresh_results([source], [fresh], [audit])


def test_partial_checker_hold_or_short_test_set_never_admitted():
    for field, value in [('input_coverage', 'partial'), ('checker_review_flags', ['tolerance']),
                         ('oracle_review_required', True), ('public_samples_checked', 0)]:
        source, fresh, audit = evidence()
        audit[field] = value
        assert not join_verified_fresh_results([source], [fresh], [audit])['eligible_ids']


def test_duplicate_source_or_changed_audit_is_rejected():
    source, fresh, audit = evidence()
    with pytest.raises(ValueError, match='duplicate_source'):
        join_verified_fresh_results([source, source], [fresh], [audit])
    audit['cleaned_tests_hash'] = '0' * 64
    with pytest.raises(ValueError, match='cleaning_audit_mismatch'):
        join_verified_fresh_results([source], [fresh], [audit])
