import pytest

from posttrain.data.schema import TestCase as Fixture
from posttrain.execution.contracts import CaseResult, Status
from posttrain.evaluation.lcb_results import adjudicate_lcb, summarize_lcb


def test_raw_duplicates_retained_and_whitespace_accepted():
    cases = (Fixture('x', '1 2'), Fixture('x', '1 2\n'))
    out = adjudicate_lcb('atcoder:synthetic', cases,
                         (CaseResult(Status.OK, '1\n2'), CaseResult(Status.OK, '1 2')))
    assert out['raw_test_count'] == 2 and out['passed'] == 2
    assert out['pass_at_1'] is True


def test_arc_checker_accepts_alternative_and_not_just_stored_text(monkeypatch):
    # Routing test, independent of real task fixtures; checker has its own proof tests.
    import posttrain.evaluation.lcb_results as mod
    seen = []
    def checker(inp, output):
        seen.append((inp, output))
        return output == 'alternative'
    monkeypatch.setattr(mod, 'check_arc188_c', checker)
    out = adjudicate_lcb('atcoder:arc188_c', (Fixture('fixture', 'stored'),),
                         (CaseResult(Status.OK, 'alternative'),))
    assert out['pass_at_1'] is True and seen == [('fixture', 'alternative')]


def test_infra_never_reduces_denominator_or_becomes_model_failure():
    out = adjudicate_lcb('synthetic', (Fixture('x', '1'),), (CaseResult(Status.INFRA),))
    assert out['pass_at_1'] is None
    with pytest.raises(ValueError, match='invalid_task_result'):
        summarize_lcb(['synthetic'], [out])


def test_missing_duplicate_and_malformed_records_rejected():
    with pytest.raises(ValueError):
        adjudicate_lcb('x', (Fixture('x','1'),), ())
    good = adjudicate_lcb('x', (Fixture('x','1'),), (CaseResult(Status.OK, '1'),))
    with pytest.raises(ValueError):
        summarize_lcb(['x','y'], [good,good])
    with pytest.raises(ValueError):
        summarize_lcb(['x','y'], [good])


def test_program_errors_count_as_failure_and_truncation_not_success():
    for result in [CaseResult(Status.TIMEOUT), CaseResult(Status.RUNTIME),
                   CaseResult(Status.OK, '1', truncated=True)]:
        out=adjudicate_lcb('x', (Fixture('x','1'),), (result,))
        assert out['pass_at_1'] is False
        assert summarize_lcb(['x'], [out])['greedy_pass_at_1'] == 0.0


def test_summary_rejects_edited_totals_or_case_statuses():
    import copy
    good = adjudicate_lcb('x', (Fixture('x','1'),), (CaseResult(Status.OK, '1'),))
    for key, value in [('passed', 0), ('pass_at_1', False), ('scorer_version', 'old')]:
        bad=copy.deepcopy(good)
        bad[key]=value
        with pytest.raises(ValueError, match='invalid_task_result'):
            summarize_lcb(['x'], [bad])
    bad=copy.deepcopy(good)
    bad['cases'][0]['status']='invented'
    with pytest.raises(ValueError, match='invalid_task_result'):
        summarize_lcb(['x'], [bad])


def test_raw_identity_changes_even_for_equivalent_expected_whitespace():
    result = (CaseResult(Status.OK, '1'),)
    one = adjudicate_lcb('x', (Fixture('x', '1'),), result)
    two = adjudicate_lcb('x', (Fixture('x', '1\n'),), result)
    assert one['pass_at_1'] == two['pass_at_1'] is True
    assert one['raw_tests_hash'] != two['raw_tests_hash']


def test_frozen_summary_binds_each_tasks_test_bytes():
    from posttrain.evaluation.lcb_results import summarize_frozen_lcb
    good = adjudicate_lcb('x', (Fixture('x','1'),), (CaseResult(Status.OK, '1'),))
    tasks = [{'task_id':'x', 'tests':(Fixture('x','1'),)}]
    assert summarize_frozen_lcb(tasks, [good])['passed_tasks'] == 1
    changed = [{'task_id':'x', 'tests':(Fixture('different','1'),)}]
    with pytest.raises(ValueError, match='raw_test_identity_mismatch'):
        summarize_frozen_lcb(changed, [good])


def test_real_arc_checker_is_used_in_final_adjudication():
    stdin = '3 3\n1 2 1\n1 3 0\n2 3 0\n'
    tests = (Fixture(stdin, '010'),)
    valid = adjudicate_lcb('atcoder:arc188_c', tests, (CaseResult(Status.OK, '110'),))
    invalid = adjudicate_lcb('atcoder:arc188_c', tests, (CaseResult(Status.OK, '000'),))
    assert valid['pass_at_1'] is True
    assert invalid['pass_at_1'] is False
