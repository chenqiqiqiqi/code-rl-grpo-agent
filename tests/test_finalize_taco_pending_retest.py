import sys
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from posttrain.common import digest
from posttrain.data.schema import Problem, TestCase as ProblemTestCase
from scripts.finalize_taco_pending_retest import (
    SOURCE_REL,
    _legacy_pair_pass_indices,
    _verify_legacy_identity,
    derive_candidate_rows,
    validate_review_partition,
)


def make_problem(problem_id, count=6):
    return Problem(
        problem_id=problem_id,
        source="synthetic",
        revision="v1",
        source_id=f"synthetic:{problem_id}",
        statement="Read an integer and print it.",
        tests=tuple(ProblemTestCase(f"{i}\n", f"{i}\n") for i in range(count)),
        references=("print(input())",),
        cluster_id=f"cluster-{problem_id}",
    ).to_dict()


def make_audit(source, *, flags=None):
    tests = source["tests"]
    return {
        "problem_id": source["problem_id"],
        "cluster_id": source["cluster_id"],
        "statement_sha256": digest(source["statement"]),
        "original_tests_hash": digest(tests),
        "retained_indices": list(range(len(tests))),
        "removed_indices": [],
        "retained_test_hashes": {str(i): digest(test) for i, test in enumerate(tests)},
        "disposition": "pending_domain_verification",
        "statement_assessment": {"status": "no_confirmed_issue", "issues": []},
        "checker_review_flags": flags or [],
        "oracle_review_required": None,
    }


def test_review_partition_derives_exact_missing_ids_and_rejects_bad_coverage():
    coverage = validate_review_partition(
        assigned_ids=["p1", "p2", "p3"],
        reviewed_ids=["p1"],
        expected_assigned=3,
        expected_reviewed=1,
        expected_unfinished=2,
    )
    assert coverage["unfinished_ids"] == ["p2", "p3"]

    with pytest.raises(ValueError, match="reviewed_id_outside_partition"):
        validate_review_partition(["p1"], ["p1", "p2"], 1, 1, 0)
    with pytest.raises(ValueError, match="direct_review_coverage_count_mismatch"):
        validate_review_partition(["p1", "p2"], ["p1"], 2, 1, 0)


def test_derive_excludes_unfinished_and_unapplied_corrections_with_lineage():
    rows = [make_problem(pid) for pid in ("p1", "p2", "p3", "p4")]
    by_id = {row["problem_id"]: row for row in rows}
    audits = {pid: make_audit(row) for pid, row in by_id.items()}
    decisions = {
        "p1": {
            "status": "provisional_repair_pending_coverage_and_fresh_backtest",
            "source_row_sha256": digest(by_id["p1"]),
            "statement_sha256": digest(by_id["p1"]["statement"]),
            "original_test_count": len(by_id["p1"]["tests"]),
            "pair_pass_indices": [0, 1, 2, 3, 4],
            "proposed_remove_indices": [5],
        },
    }
    corrections = [{
        "problem_id": "p3",
        "source_row_sha256": digest(by_id["p3"]),
        "statement_sha256": digest(by_id["p3"]["statement"]),
        "defect_type": "statement_or_source_correction",
        "repair_applied": False,
    }]
    overrides = [{
        "problem_id": "p4",
        "source_row_sha256": digest(by_id["p4"]),
        "statement_sha256": digest(by_id["p4"]["statement"]),
        "disposition": "quarantine_unsupported_checker",
    }]

    result = derive_candidate_rows(
        source_rows=rows,
        base_audits=audits,
        base_cleaned_rows=by_id,
        prior_reviewed_ids={"p4"},
        direct_partition_ids={"p1", "p2", "p3"},
        direct_decisions=decisions,
        corrections=corrections,
        overrides=overrides,
        unfinished_ids={"p2", "p3"},
    )

    assert [row["problem_id"] for row in result["problems"]] == ["p1"]
    assert len(result["problems"][0]["tests"]) == 5
    assert result["problems"][0]["validation_status"] == "pending_isolated_backtest"
    lineage = {row["problem_id"]: row for row in result["lineage"]}
    assert lineage["p1"]["included"] is True
    assert lineage["p1"]["retained_source_test_indices"] == [0, 1, 2, 3, 4]
    assert lineage["p2"]["exclusion_reasons"] == ["unfinished_direct_review"]
    assert lineage["p3"]["exclusion_reasons"] == [
        "source_correction_unapplied", "unfinished_direct_review"
    ]
    assert "direct_disposition_override" in lineage["p4"]["exclusion_reasons"]


def test_derivation_keeps_source_and_predecessor_rows_unchanged():
    source = make_problem("p1")
    audit = make_audit(source)
    decision = {
        "status": "provisional_repair_pending_coverage_and_fresh_backtest",
        "source_row_sha256": digest(source),
        "statement_sha256": digest(source["statement"]),
        "original_test_count": len(source["tests"]),
        "pair_pass_indices": list(range(len(source["tests"]))),
        "proposed_remove_indices": [],
    }
    source_before = deepcopy(source)
    audit_before = deepcopy(audit)
    derive_candidate_rows(
        source_rows=[source],
        base_audits={"p1": audit},
        base_cleaned_rows={"p1": source},
        prior_reviewed_ids=set(),
        direct_partition_ids={"p1"},
        direct_decisions={"p1": decision},
        corrections=[],
        overrides=[],
        unfinished_ids=set(),
    )
    assert source == source_before
    assert audit == audit_before


def test_derive_binds_source_hash_and_rejects_missing_decision_coverage():
    source = make_problem("p1")
    audit = make_audit(source)
    decision = {
        "status": "provisional_repair_pending_coverage_and_fresh_backtest",
        "source_row_sha256": "0" * 64,
        "statement_sha256": digest(source["statement"]),
        "original_test_count": len(source["tests"]),
        "pair_pass_indices": list(range(len(source["tests"]))),
        "proposed_remove_indices": [],
    }
    kwargs = dict(
        source_rows=[source],
        base_audits={"p1": audit},
        base_cleaned_rows={"p1": source},
        prior_reviewed_ids=set(),
        direct_partition_ids={"p1"},
        direct_decisions={"p1": decision},
        corrections=[],
        overrides=[],
        unfinished_ids=set(),
    )
    with pytest.raises(ValueError, match="direct_decision_source_mismatch:p1"):
        derive_candidate_rows(**kwargs)

    kwargs["direct_decisions"] = {}
    with pytest.raises(ValueError, match="direct_decision_coverage_mismatch"):
        derive_candidate_rows(**kwargs)


def test_per_problem_checker_hold_is_excluded_but_global_pending_state_is_not():
    candidate = make_problem("p1")
    blocked = make_problem("p2")
    audits = {
        "p1": make_audit(candidate),
        "p2": make_audit(blocked, flags=[{"reason": "semantic_checker_required"}]),
    }
    result = derive_candidate_rows(
        source_rows=[candidate, blocked],
        base_audits=audits,
        base_cleaned_rows={"p1": candidate, "p2": blocked},
        prior_reviewed_ids={"p1", "p2"},
        direct_partition_ids=set(),
        direct_decisions={},
        corrections=[],
        overrides=[],
        unfinished_ids=set(),
    )
    assert [row["problem_id"] for row in result["problems"]] == ["p1"]
    lineage = {row["problem_id"]: row for row in result["lineage"]}
    assert lineage["p1"]["exclusion_reasons"] == []
    assert "base_checker_hold" in lineage["p2"]["exclusion_reasons"]


def test_legacy_direct_reviews_keep_only_double_pass_tests_and_hold_unknowns():
    fixed_row = {"tests": [
        {"input_conclusion": "valid", "expected_output_conclusion": "correct"},
        {"input_conclusion": "invalid", "expected_output_conclusion": "unassessable_invalid_input"},
        {"input_conclusion": "valid", "expected_output_conclusion": "incorrect"},
    ]}
    assert _legacy_pair_pass_indices("fixed", fixed_row) == [0]
    assert _legacy_pair_pass_indices("fixed", {"tests": [
        {"input_conclusion": "unknown", "expected_output_conclusion": "correct"}
    ]}) is None

    passing = make_problem("p1", count=7)
    uncertain = make_problem("p2", count=7)
    audits = {pid: make_audit(row) for pid, row in (("p1", passing), ("p2", uncertain))}
    audits["p1"]["retained_indices"] = [0, 1, 2, 3, 4, 5]
    audits["p1"]["retained_test_hashes"] = {
        str(i): digest(passing["tests"][i]) for i in range(6)
    }
    passing_v3 = deepcopy(passing)
    passing_v3["tests"] = passing["tests"][:6]
    result = derive_candidate_rows(
        source_rows=[passing, uncertain],
        base_audits=audits,
        base_cleaned_rows={"p1": passing_v3, "p2": uncertain},
        prior_reviewed_ids={"p1", "p2"},
        direct_partition_ids=set(),
        direct_decisions={},
        corrections=[],
        overrides=[],
        unfinished_ids=set(),
        prior_reviewed_tests_by_id={"p1": [0, 1, 2, 3, 4, 6], "p2": None},
    )
    assert [row["problem_id"] for row in result["problems"]] == ["p1"]
    assert [test["stdin"] for test in result["problems"][0]["tests"]] == [
        f"{i}\n" for i in range(5)
    ]
    lineage = {row["problem_id"]: row for row in result["lineage"]}
    assert lineage["p1"]["retained_source_test_indices"] == [0, 1, 2, 3, 4]
    assert lineage["p2"]["exclusion_reasons"] == ["legacy_direct_review_uncertain"]


def test_statement_quarantine_without_retained_test_index_is_held():
    source = make_problem("p1")
    audit = make_audit(source)
    audit["disposition"] = "statement_quarantine"
    audit.pop("retained_indices")
    audit.pop("retained_test_hashes")
    result = derive_candidate_rows(
        source_rows=[source],
        base_audits={"p1": audit},
        base_cleaned_rows={},
        prior_reviewed_ids={"p1"},
        direct_partition_ids=set(),
        direct_decisions={},
        corrections=[],
        overrides=[],
        unfinished_ids=set(),
    )
    assert result["problems"] == []
    assert "base_statement_quarantine" in result["lineage"][0]["exclusion_reasons"]
    assert "base_cleaning_quarantine" in result["lineage"][0]["exclusion_reasons"]


def test_legacy_review_checkpoint_binds_source_statement_and_each_test():
    source = make_problem("p1", count=1)
    test = source["tests"][0]
    identity = {
        "line_index": 4,
        "raw_line_sha256": "a" * 64,
        "source_id": source["source_id"],
        "statement_sha256_utf8": hashlib.sha256(
            source["statement"].encode("utf-8")).hexdigest(),
        "tests": [{
            "canonical_sha256": digest(test),
            "stdin_sha256": hashlib.sha256(test["stdin"].encode()).hexdigest(),
            "stdout_sha256": hashlib.sha256(test["stdout"].encode()).hexdigest(),
            "stdin_nul_stdout_sha256": hashlib.sha256(
                (test["stdin"] + "\0" + test["stdout"]).encode()).hexdigest(),
        }],
    }
    fixed = {
        "problem_id": "p1", "source_row_sha256": "a" * 64,
        "statement_sha256_utf8": identity["statement_sha256_utf8"],
        "review_status": "complete", "all_tests_reviewed": True,
        "test_count_total": 1, "test_indices_reviewed": [0],
        "tests": [{"test_index": 0, "test_sha256_canonical_json": digest(test)}],
    }
    for completed_status in (
        "complete",
        "complete_with_invalid_test_inputs",
        "complete_with_expected_output_errors",
    ):
        fixed["review_status"] = completed_status
        _verify_legacy_identity("fixed", fixed, identity)
    fixed["review_status"] = "incomplete"
    with pytest.raises(ValueError, match="prior_checkpoint_incomplete:p1"):
        _verify_legacy_identity("fixed", fixed, identity)
    fixed["review_status"] = "complete"
    fixed["tests"][0]["test_sha256_canonical_json"] = "b" * 64
    with pytest.raises(ValueError, match="prior_test_hash_mismatch:p1"):
        _verify_legacy_identity("fixed", fixed, identity)

    arrays = {
        "problem_id": "p1", "source_id": source["source_id"],
        "statement": {"full_statement_read": True,
                      "source_statement_sha256": identity["statement_sha256_utf8"]},
        "actual_coverage": {"tests_fully_read": 1, "tests_total": 1,
                             "test_indices_zero_based": [0]},
        "tests": [{"test_index_zero_based": 0, "stdin_sha256": identity["tests"][0]["stdin_sha256"],
                   "stdout_sha256": identity["tests"][0]["stdout_sha256"]}],
    }
    _verify_legacy_identity("arrays", arrays, identity)

    rows = {
        "problem_id": "p1", "source_id": source["source_id"],
        "review_status": "complete_direct_question_and_all_physical_tests",
        "problem_source": {"path": SOURCE_REL.as_posix(), "line_zero_based": 4,
                           "row_sha256": "a" * 64},
        "tests": [{"test_index": 0, "test_identity": {
            "problem_id": "p1", "test_index": 0,
            "sha256_stdin_nul_stdout_utf8": identity["tests"][0]["stdin_nul_stdout_sha256"]}}],
    }
    _verify_legacy_identity("rows", rows, identity)
