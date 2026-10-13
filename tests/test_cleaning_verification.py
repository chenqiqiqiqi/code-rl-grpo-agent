from dataclasses import replace
import pytest
from posttrain.common import digest
from posttrain.data.selective_repair import repair_problem
from posttrain.data.cleaning_verification import verify_lineage


def fixture(problem):
    cleaned,audit=repair_problem(problem,[])
    audit.update(statement_sha256=digest(problem.statement),training_ready=False,
        disposition='pending_domain_verification',historical_reference_evidence={
            'new_executions':0,'fresh_reference_backtest':False,'retained_all_pass':False})
    return cleaned,audit


def test_valid_lineage_passes_without_running_compiler(problem):
    cleaned,audit=fixture(problem)
    verify_lineage(problem,audit,cleaned)


def test_changed_answer_or_statement_cannot_hide_behind_report(problem):
    cleaned,audit=fixture(problem)
    changed=replace(cleaned,tests=(replace(cleaned.tests[0],stdout='tampered'),)+cleaned.tests[1:])
    with pytest.raises(ValueError,match='retained_tests_changed'):verify_lineage(problem,audit,changed)
    with pytest.raises(ValueError,match='changed_problem_field'):verify_lineage(problem,audit,replace(cleaned,statement='tampered'))


def test_missing_case_and_inflated_status_are_rejected(problem):
    cleaned,audit=fixture(problem)
    with pytest.raises(ValueError,match='unproven_validation_status'):
        verify_lineage(problem,audit,replace(cleaned,validation_status='reference_passed'))
    audit['retained_indices']=audit['retained_indices'][:-1]
    with pytest.raises(ValueError,match='retained_partition_mismatch'):verify_lineage(problem,audit,cleaned)
