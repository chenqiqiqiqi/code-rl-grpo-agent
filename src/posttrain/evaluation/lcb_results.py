"""Final stdio adjudication of already executed cases; never executes code.

This does not grant permission/readiness for final model evaluation. Results must
come from an independently attested execution pipeline with a frozen manifest.
All raw cases remain present, including repeated inputs. Training Judge unchanged.
"""
from __future__ import annotations

from posttrain.common import digest
from posttrain.data.schema import TestCase, normalize_output
from posttrain.execution.contracts import CaseResult, Status

SCORER_VERSION = 'lcb-project-arc188-checker-v1'


def check_arc188_c(stdin: str, stdout: str) -> bool:
    from posttrain.evaluation.lcb_checker import check_arc188_c as check
    return check(stdin, stdout)


def adjudicate_lcb(task_id: str, tests: tuple[TestCase, ...],
                   results: tuple[CaseResult, ...]) -> dict:
    """Require a result for every raw fixture, with task-specific comparison."""
    if not task_id or not tests or len(tests) != len(results):
        raise ValueError('incomplete_raw_case_results')
    verdicts = []
    for index, (test, result) in enumerate(zip(tests, results)):
        if not isinstance(test, TestCase) or not isinstance(result, CaseResult):
            raise ValueError('malformed_case_record')
        if not isinstance(result.status, Status):
            raise ValueError('unknown_execution_status')
        status = result.status
        if status == Status.OK:
            if result.truncated:
                status = Status.OUTPUT
            else:
                matches = (check_arc188_c(test.stdin, result.stdout)
                           if task_id == 'atcoder:arc188_c' else
                           normalize_output(test.stdout) == normalize_output(result.stdout))
                if not matches:
                    status = Status.WRONG
        verdicts.append({'original_test_index': index, 'status': status.value})
    invalid = any(v['status'] in (Status.INFRA.value, Status.DATA.value) for v in verdicts)
    passed = sum(v['status'] == Status.OK.value for v in verdicts)
    return {'task_id': task_id, 'scorer_version': SCORER_VERSION,
            'raw_tests_hash': digest([{'stdin': t.stdin, 'stdout': t.stdout} for t in tests]),
            'raw_test_count': len(tests), 'passed': passed, 'valid': not invalid,
            'pass_at_1': None if invalid else passed == len(tests), 'cases': verdicts}


def summarize_lcb(frozen_task_ids: list[str], rows: list[dict]) -> dict:
    """Pause aggregation on any missing/duplicate task or invalid execution."""
    ids = [r['task_id'] for r in rows]
    if (not frozen_task_ids or len(set(frozen_task_ids)) != len(frozen_task_ids)
            or len(ids) != len(set(ids)) or set(ids) != set(frozen_task_ids)):
        raise ValueError('frozen_task_coverage_mismatch')
    for row in rows:
        cases = row.get('cases', [])
        statuses = [v.get('status') for v in cases]
        passed = sum(s == Status.OK.value for s in statuses)
        if (row.get('scorer_version') != SCORER_VERSION or row.get('valid') is not True
                or type(row.get('pass_at_1')) is not bool
                or row.get('raw_test_count', 0) <= 0
                or len(cases) != row['raw_test_count']
                or [v.get('original_test_index') for v in cases] != list(range(len(cases)))
                or any(s not in {x.value for x in Status} or s in
                       (Status.DATA.value, Status.INFRA.value) for s in statuses)
                or row.get('passed') != passed
                or row['pass_at_1'] != (passed == len(cases))):
            raise ValueError('invalid_task_result')
    total = len(frozen_task_ids)
    passed = sum(r['pass_at_1'] for r in rows)
    return {'task_count': total, 'passed_tasks': passed,
            'greedy_pass_at_1': passed / total, 'scorer_version': SCORER_VERSION}


def summarize_frozen_lcb(tasks: list[dict], rows: list[dict]) -> dict:
    """Use verified loader output to reject scores for stale/different tests.

    The integer/boolean checks in summarize_lcb verify record consistency, not
    provenance. The execution runner must separately authenticate its receipts.
    """
    summary = summarize_lcb([t['task_id'] for t in tasks], rows)
    expected = {t['task_id']: digest([{'stdin': c.stdin, 'stdout': c.stdout}
                                    for c in t['tests']]) for t in tasks}
    if any(row.get('raw_tests_hash') != expected[row['task_id']] for row in rows):
        raise ValueError('raw_test_identity_mismatch')
    summary['raw_tests_hashes'] = expected
    return summary
