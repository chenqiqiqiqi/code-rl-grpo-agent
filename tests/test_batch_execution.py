from posttrain.data.schema import TestCase as Case
from posttrain.execution.contracts import CaseResult, Limits, Status
from posttrain.execution.judge import Judge


class BatchBackend:
    version = 'batch-contract-v1'
    evidence_level = 'mock_contract'

    def __init__(self, initial, retries=()):
        self.initial = initial
        self.retries = iter(retries)
        self.batch_calls = []
        self.single_calls = []

    def run_batch(self, code, stdins, limits, request_id, timeout_s):
        self.batch_calls.append((code, stdins))
        return self.initial

    def run(self, code, stdin, limits, request_id, timeout_s):
        self.single_calls.append(stdin)
        return next(self.retries)


def test_batch_scores_every_case_and_passes_no_expected_answers():
    backend = BatchBackend([CaseResult(Status.OK, stdout='yes'), CaseResult(Status.RUNTIME)])
    result = Judge(backend).score('code', (Case('a', 'yes'), Case('b', 'hidden')), Limits())
    assert backend.batch_calls == [('code', ['a', 'b'])]
    assert result.valid and result.passed == 1 and result.physical_executions == 2


def test_batch_accounts_all_physical_work_even_after_invalid_result():
    bad = CaseResult(Status.INFRA, physical_executions=1)
    backend = BatchBackend([bad, CaseResult(Status.OK, stdout='yes')], [bad, bad])
    result = Judge(backend, backoff_s=0).score('code', (Case('a', 'yes'), Case('b', 'yes')), Limits())
    assert not result.valid and result.physical_executions == 4 and result.retries == 2
    assert len(result.cases) == 2


def test_retry_budget_is_per_candidate_not_per_test():
    bad = CaseResult(Status.INFRA, physical_executions=1)
    good = CaseResult(Status.OK, stdout='yes')
    backend = BatchBackend([bad, bad, bad], [good, good])
    result = Judge(backend, backoff_s=0).score('code', tuple(Case(str(i), 'yes') for i in range(3)), Limits())
    assert not result.valid and result.retries == 2 and result.physical_executions == 5
    assert backend.single_calls == ['0', '1']


def test_malformed_batch_cannot_silently_drop_tests():
    backend = BatchBackend([CaseResult(Status.OK, stdout='yes')])
    result = Judge(backend, max_retries=0).score('code', (Case('a', 'yes'), Case('b', 'yes')), Limits())
    assert not result.valid and result.physical_executions is None


def test_group_rejects_mixed_test_identities():
    from dataclasses import replace
    from posttrain.training.groups import CandidateGroup
    backend=BatchBackend([CaseResult(Status.OK,stdout='yes')])
    result=Judge(backend).score('code',(Case('a','yes'),),Limits())
    assert not CandidateGroup('p',(2,2),(result,replace(result,tests_hash='wrong-problem'))).valid(2)


def test_deadline_admission_happens_before_any_execution():
    import pytest
    backend=BatchBackend([])
    with pytest.raises(ValueError,match='deadline_cannot_cover_full_tests'):
        Judge(backend,deadline_s=3).score('code',(Case('a','yes'),Case('b','yes')),Limits(time_s=2))
    assert backend.batch_calls==[]
