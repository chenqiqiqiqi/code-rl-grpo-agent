"""Binary short circuit may omit cases only after a decisive model failure."""
from dataclasses import replace
import threading
import urllib.error

import pytest

from posttrain.data.schema import TestCase as Case
from posttrain.execution.contracts import CaseResult, Limits, Status
from posttrain.execution.judge import Judge
from posttrain.rewards import rewards


class BatchBackend:
    version = 'isolated-fixture-v1'
    evidence_level = 'isolated_execution'

    def __init__(self, rows):
        self.rows = rows
        self.dispatched = []

    def run_batch(self, code, stdins, limits, request_id, timeout_s):
        self.dispatched.append(list(stdins))
        return [self.rows[int(stdin)] for stdin in stdins]

    def run(self, code, stdin, limits, request_id, timeout_s):
        return self.rows[int(stdin)]


def cases(n):
    return tuple(Case(str(i), 'ok') for i in range(n))


def ok():
    return CaseResult(Status.OK, stdout='ok', queue_s=0)


def test_first_failure_stops_after_one_real_case_and_keeps_frozen_denominator():
    backend = BatchBackend([replace(ok(), status=Status.TIMEOUT), *[ok() for _ in range(19)]])
    result = Judge(backend).score('x', cases(20), Limits(), 'first', scoring_mode='binary_fail_fast')
    assert backend.dispatched == [['0']]
    assert result.binary_valid and not result.valid
    assert (result.passed, result.total, len(result.cases), result.physical_executions) == (0, 20, 1, 1)
    assert result.to_dict()['skipped_case_judgments'] == 19
    assert rewards(result) is None
    assert rewards(result, binary_only=True) == {'binary': 0.0}


def test_later_failure_consumes_dispatched_chunk_and_same_binary_as_full():
    rows = [ok() for _ in range(20)]
    rows[3] = replace(ok(), stdout='wrong')
    full_backend, fast_backend = BatchBackend(rows), BatchBackend(rows)
    full = Judge(full_backend).score('x', cases(20), Limits(), 'full')
    fast = Judge(fast_backend).score('x', cases(20), Limits(), 'fast', scoring_mode='binary_fail_fast')
    assert len(full.cases) == 20 and len(fast.cases) == 9
    assert fast_backend.dispatched == [['0'], [str(i) for i in range(1, 9)]]
    assert fast.binary_valid and not fast.valid
    assert rewards(full)['binary'] == rewards(fast, binary_only=True)['binary'] == 0.0


def test_full_pass_still_executes_every_frozen_case():
    backend = BatchBackend([ok() for _ in range(20)])
    result = Judge(backend).score('x', cases(20), Limits(), scoring_mode='binary_fail_fast')
    assert result.valid and result.binary_valid and result.passed == result.total == 20
    assert sum(map(len, backend.dispatched)) == 20
    assert rewards(result, binary_only=True) == {'binary': 1.0}


def test_later_infrastructure_error_prevents_binary_zero_after_earlier_wrong():
    rows = [ok() for _ in range(20)]
    rows[3] = replace(ok(), stdout='wrong')
    rows[7] = CaseResult(Status.INFRA, physical_executions=None, detail='container_failed')
    backend = BatchBackend(rows)
    result = Judge(backend, max_retries=0).score('x', cases(20), Limits(),
                                                  scoring_mode='binary_fail_fast')
    assert len(result.cases) == 9 and result.status == Status.INFRA
    assert not result.binary_valid and rewards(result, binary_only=True) is None


def test_cache_is_separate_from_full_scoring_and_no_fabricated_work():
    backend = BatchBackend([replace(ok(), stdout='wrong'), *[ok() for _ in range(19)]])
    judge = Judge(backend)
    fast = judge.score('x', cases(20), Limits(), 'a', scoring_mode='binary_fail_fast')
    cached = judge.score('x', cases(20), Limits(), 'b', scoring_mode='binary_fail_fast')
    full = judge.score('x', cases(20), Limits(), 'c')
    assert fast.binary_valid and cached.cache_hit and cached.physical_executions == 0
    assert len(cached.cases) == 1 and len(full.cases) == 20
    assert sum(map(len, backend.dispatched)) == 21


def test_mixed_setting_cannot_enable_binary_short_circuit():
    from posttrain.training.code_rl import checked_settings
    with pytest.raises(ValueError, match='binary_fail_fast_requires_binary_reward'):
        checked_settings({'mode':'single','reward_kind':'mixed','scoring_mode':'binary_fail_fast'})


def test_http_receipt_binds_mode_and_reconciles_without_redispatch(tmp_path, problem):
    from posttrain.execution.service import ScoringClient, make_server
    frozen = replace(problem, validation_status='reference_passed',
                     tests=cases(20))
    backend = BatchBackend([replace(ok(), stdout='wrong'), *[ok() for _ in range(19)]])
    server = make_server([frozen], backend, tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    expected = {frozen.problem_id:{'tests_hash':frozen.tests_hash,'test_count':20}}
    endpoint = f'http://127.0.0.1:{server.server_port}'
    fast = ScoringClient(endpoint, backend.version, expected, scoring_mode='binary_fail_fast')
    full = ScoringClient(endpoint, backend.version, expected)
    try:
        result = fast.score(frozen.problem_id, 'x', 'one')
        assert result.binary_valid and not result.valid and len(result.cases)==1
        assert len(backend.dispatched)==1
        assert fast.reconcile(frozen.problem_id,'x','one')==result
        with pytest.raises(ValueError,match='mode_mismatch'):
            full.reconcile(frozen.problem_id,'x','one')
        with pytest.raises(urllib.error.HTTPError) as exc:
            fast.score(frozen.problem_id,'x','one')
        assert exc.value.code==409 and len(backend.dispatched)==1
    finally:
        server.shutdown();server.server_close();worker.join()
