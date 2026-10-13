from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from posttrain.common import digest, file_digest, write_json
from posttrain.data.schema import Problem
from posttrain.data.selective_repair import historical_subset_evidence
from scripts import merge_direct_review_pool as merge_module
from scripts import verify_merged_direct_pool as verifier_module
from scripts import build_direct_review_decisions
from scripts.merge_direct_review_pool import (
    PINNED_V3_BASE_REPORT_SHA256,
    PINNED_PARENT_RUN as MERGE_PINNED_PARENT_RUN,
    _baseline_quality_blockers,
    _validate_source_correction_gates,
    _validate_pinned_parent_path as merge_validate_pinned_parent_path,
    _validate_pinned_v3_base,
)
from scripts.verify_merged_direct_pool import (
    PINNED_PARENT_RUN as VERIFY_PINNED_PARENT_RUN,
    PINNED_V3_BASE_REPORT_SHA256 as VERIFY_PINNED_V3_BASE_REPORT_SHA256,
    _baseline_quality_blockers as verifier_baseline_quality_blockers,
    _expected_direct_gate,
    _recompute_direct_metrics,
    _validate_reproducible_decisions,
    _validate_source_correction_gates as verifier_validate_source_correction_gates,
    _validate_pinned_parent_path as verifier_validate_pinned_parent_path,
    _validate_pinned_v3_base as verifier_validate_pinned_v3_base,
    _validate_historical_subset_proof,
    _validate_statement_quarantine_lineage,
)


def test_pinned_base_requires_v3_run_id_and_report_hash(tmp_path, monkeypatch):
    assert VERIFY_PINNED_V3_BASE_REPORT_SHA256 == PINNED_V3_BASE_REPORT_SHA256
    report_path = tmp_path / "report.json"
    write_json(report_path, {
        "run_id": "stage1-completion-cleaning-local-v3",
        "scope": "full_3082_candidate_pool_static_cleaning",
    })
    monkeypatch.setattr(
        "scripts.merge_direct_review_pool.PINNED_V3_BASE_REPORT_SHA256",
        file_digest(report_path),
    )
    expected = {
        "run_id": "stage1-completion-cleaning-local-v3",
        "scope": "full_3082_candidate_pool_static_cleaning",
    }
    _validate_pinned_v3_base(tmp_path, expected)
    monkeypatch.setattr(
        "scripts.verify_merged_direct_pool.PINNED_V3_BASE_REPORT_SHA256",
        file_digest(report_path),
    )
    verifier_validate_pinned_v3_base(tmp_path, expected)

    with pytest.raises(ValueError, match="pinned_base_run_id"):
        _validate_pinned_v3_base(tmp_path, {"run_id": "another-run"})

    monkeypatch.setattr(
        "scripts.merge_direct_review_pool.PINNED_V3_BASE_REPORT_SHA256",
        PINNED_V3_BASE_REPORT_SHA256,
    )
    monkeypatch.setattr(
        "scripts.verify_merged_direct_pool.PINNED_V3_BASE_REPORT_SHA256",
        VERIFY_PINNED_V3_BASE_REPORT_SHA256,
    )
    with pytest.raises(ValueError, match="pinned_base_report_sha256"):
        _validate_pinned_v3_base(tmp_path, expected)
    with pytest.raises(ValueError, match="pinned_base_report_sha256"):
        verifier_validate_pinned_v3_base(tmp_path, expected)


def test_parent_run_must_be_the_fixed_trusted_local_path(tmp_path):
    assert MERGE_PINNED_PARENT_RUN == VERIFY_PINNED_PARENT_RUN
    merge_validate_pinned_parent_path(MERGE_PINNED_PARENT_RUN)
    verifier_validate_pinned_parent_path(VERIFY_PINNED_PARENT_RUN)
    with pytest.raises(ValueError, match="pinned_parent_run_path_mismatch"):
        merge_validate_pinned_parent_path(tmp_path)
    with pytest.raises(ValueError, match="pinned_parent_run_path_mismatch"):
        verifier_validate_pinned_parent_path(tmp_path)


def test_rebuilt_decision_manifest_rejects_checkpoint_decision_tampering():
    decisions = [{"problem_id": f"p{i:04d}", "status": "hold_unresolved_test"}
                 for i in range(1833)]
    document = {
        "checkpoint_sha256": {f"direct_review_shard_{i:02d}_checkpoint_v2.json": "a" * 64
                              for i in range(1, 21)},
        "complete_direct_review": True,
        "reviewed_problem_count": 1833,
        "decisions": decisions,
        "training_ready": False,
        "source_sha256": "b" * 64,
    }
    _validate_reproducible_decisions(document, deepcopy(document), "b" * 64)

    tampered = deepcopy(document)
    tampered["decisions"][0]["status"] = "provisional_repair_pending_coverage_and_fresh_backtest"
    with pytest.raises(ValueError, match="direct_decisions_not_reproducible_from_checkpoints"):
        _validate_reproducible_decisions(tampered, document, "b" * 64)


def test_baseline_quality_blockers_fail_closed_for_provisional_candidates():
    clean = {
        "statement_assessment": {"status": "no_confirmed_issue", "issues": []},
        "checker_review_flags": [],
        "oracle_review_required": None,
    }
    assert _baseline_quality_blockers(clean) == []
    assert verifier_baseline_quality_blockers(clean) == []

    blocked = {
        "statement_assessment": {"status": "review_needed", "issues": ["unclear"]},
        "checker_review_flags": [{"reason": "custom_checker"}],
        "oracle_review_required": True,
    }
    expected = ["statement_assessment", "checker_review_flags", "oracle_review_required"]
    assert _baseline_quality_blockers(blocked) == expected
    assert verifier_baseline_quality_blockers(blocked) == expected
    assert _baseline_quality_blockers({}) == ["statement_assessment", "checker_review_flags"]


def test_source_correction_status_cannot_be_overridden_by_checkpoint_decision():
    correction = [{"problem_id": "taco-train-008078"}]
    forged_complete = {"taco-train-008078": {"status": "provisional_repair_pending_coverage_and_fresh_backtest"}}
    with pytest.raises(ValueError, match="source_correction_status_not_held"):
        _validate_source_correction_gates(forged_complete, correction, [])
    with pytest.raises(ValueError, match="source_correction_status_not_held"):
        verifier_validate_source_correction_gates(forged_complete, correction, [])

    valid_hold = {"taco-train-008078": {"status": "hold_source_statement_repair"}}
    _validate_source_correction_gates(valid_hold, correction, [])
    verifier_validate_source_correction_gates(valid_hold, correction, [])

    pair_binding_correction = [{
        "problem_id": "taco-train-004976",
        "defect_type": "test_pair_binding",
    }]
    pair_binding_hold = {
        "taco-train-004976": {"status": "hold_corrupt_test_binding"}
    }
    _validate_source_correction_gates(pair_binding_hold, pair_binding_correction, [])
    verifier_validate_source_correction_gates(pair_binding_hold, pair_binding_correction, [])
    with pytest.raises(ValueError, match="source_correction_status_not_held"):
        _validate_source_correction_gates(valid_hold, pair_binding_correction, [])
    with pytest.raises(ValueError, match="source_correction_status_not_held"):
        verifier_validate_source_correction_gates(valid_hold, pair_binding_correction, [])

    unsupported_correction = [{
        "problem_id": "taco-train-004976",
        "defect_type": "unknown",
    }]
    with pytest.raises(ValueError, match="unsupported_source_correction_defect_type"):
        _validate_source_correction_gates(pair_binding_hold, unsupported_correction, [])
    with pytest.raises(ValueError, match="unsupported_source_correction_defect_type"):
        verifier_validate_source_correction_gates(pair_binding_hold, unsupported_correction, [])


def test_test_pair_binding_correction_builds_nonpruning_hold(tmp_path, monkeypatch):
    root = tmp_path
    (root / "reports").mkdir()
    (root / "manifests").mkdir()
    source = root / "problems.jsonl"
    row = {
        "problem_id": "pair-binding-p1",
        "source": "synthetic",
        "revision": "v1",
        "source_id": "synthetic:pair-binding",
        "statement": "Read n and print n plus one.",
        "tests": [
            {"stdin": "1\\n", "stdout": "2\\n"},
            {"stdin": "2\\n", "stdout": "3\\n"},
        ],
        "references": ["reference source is evidence only"],
        "difficulty": "easy",
        "interface": "stdio",
        "cluster_id": "cluster-pair-binding-p1",
        "validation_status": "unverified",
        "metadata": {},
    }
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(root / "manifests/direct_review_policy_v3.json", {"policy": "synthetic"})
    write_json(root / "manifests/direct_review_runtime_override_v1.json", {"runtime": "synthetic"})
    write_json(root / "reports/stage1_direct_disposition_overrides_v1.json", {
        "dispositions": [],
    })
    write_json(root / "reports/stage1_direct_source_corrections_v1.json", {
        "corrections": [{
            "problem_id": row["problem_id"],
            "source_row_sha256": digest(row),
            "statement_sha256": digest(row["statement"]),
            "defect_type": "test_pair_binding",
        }],
    })
    monkeypatch.setattr(build_direct_review_decisions, "summarize", lambda _: {
        "checkpoint_verification": {
            "invalid_reviews": [],
            "shards_without_checkpoint": [],
            "source_sha256": file_digest(source),
            "partition_manifest_sha256": "a" * 64,
        },
        "problems": [{
            "problem_id": row["problem_id"],
            "source_row_sha256": digest(row),
            "statement_sha256": digest(row["statement"]),
            "test_count": len(row["tests"]),
            "pair_pass_indices": [0],
            "input_fail_indices": [],
            "output_fail_on_legal_indices": [1],
            "uncertain_on_legal_indices": [],
            "uncertain_input_indices": [],
        }],
    })

    decision = build_direct_review_decisions.build(root, source)["decisions"][0]

    assert decision["status"] == "hold_corrupt_test_binding"
    assert decision["proposed_remove_indices"] == []
    assert decision["withheld_nonpass_indices"] == [1]
    assert decision["eligible_for_provisional_test_pruning"] is False
    assert decision["proposal_applied"] is False
    assert decision["training_ready"] is False
    assert _expected_direct_gate(
        {"disposition": "pending_domain_verification"}, decision, False
    ) == "hold_corrupt_test_binding"


def test_unsupported_interface_override_builds_exact_quarantine_status(tmp_path, monkeypatch):
    root = tmp_path
    (root / "reports").mkdir()
    (root / "manifests").mkdir()
    source = root / "problems.jsonl"
    row = {
        "problem_id": "interactive-p1",
        "source": "synthetic",
        "revision": "v1",
        "source_id": "cf:1270D",
        "statement": "This problem is interactive.\nInput protocol: query-response.",
        "tests": [{"stdin": "n k m\\n", "stdout": "? 1\\n! 1\\n"}],
        "references": ["reference source is evidence only"],
        "difficulty": "medium",
        "interface": "stdio",
        "cluster_id": "cluster-interactive-p1",
        "validation_status": "unverified",
        "metadata": {},
    }
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(root / "manifests/direct_review_policy_v3.json", {"policy": "synthetic"})
    write_json(root / "manifests/direct_review_runtime_override_v1.json", {"runtime": "synthetic"})
    write_json(root / "reports/stage1_direct_disposition_overrides_v1.json", {
        "dispositions": [{
            "problem_id": row["problem_id"],
            "source_row_sha256": digest(row),
            "statement_sha256": digest(row["statement"]),
            "disposition": "quarantine_unsupported_interface",
        }],
    })
    write_json(root / "reports/stage1_direct_source_corrections_v1.json", {"corrections": []})
    monkeypatch.setattr(build_direct_review_decisions, "summarize", lambda _: {
        "checkpoint_verification": {
            "invalid_reviews": [],
            "shards_without_checkpoint": [],
            "source_sha256": file_digest(source),
            "partition_manifest_sha256": "a" * 64,
        },
        "problems": [{
            "problem_id": row["problem_id"],
            "source_row_sha256": digest(row),
            "statement_sha256": digest(row["statement"]),
            "test_count": len(row["tests"]),
            "pair_pass_indices": [0],
            "input_fail_indices": [],
            "output_fail_on_legal_indices": [],
            "uncertain_on_legal_indices": [],
            "uncertain_input_indices": [],
        }],
    })

    decision_doc = build_direct_review_decisions.build(root, source)
    decision = decision_doc["decisions"][0]

    assert decision["status"] == "quarantine_unsupported_interface"
    assert decision["proposal_applied"] is False
    assert decision["proposed_remove_indices"] == []
    assert _expected_direct_gate(
        {"disposition": "pending_domain_verification"}, decision, False
    ) == "quarantine_unsupported_interface"


def test_interface_quarantine_preserves_test_lineage_without_reclassifying_statement(problem):
    original_statement_assessment = {"status": "no_confirmed_issue", "issues": []}
    audit = {
        "problem_id": problem.problem_id,
        "disposition": "statement_quarantine",
        "statement_assessment": deepcopy(original_statement_assessment),
    }

    merge_module._interface_quarantine_lineage(problem, audit)

    assert audit["disposition"] == "statement_quarantine"
    assert audit["statement_assessment"] == original_statement_assessment
    assert audit["interface_assessment"] == {
        "status": "unsupported_for_current_one_shot_stdio_judge",
        "protocol": "interactive",
        "reason": "interactive_query_response_protocol_is_not_supported_by_current_one_shot_stdio_judge",
    }
    assert audit["removed_indices"] == []
    assert audit["removed_test_hashes"] == {}
    assert audit["retained_indices"] == list(range(len(problem.tests)))
    assert audit["retained_test_hashes"] == {
        str(i): digest(asdict(test)) for i, test in enumerate(problem.tests)
    }
    assert audit["retained_test_count"] == len(problem.tests)
    assert audit["removed_test_count"] == 0

    verifier_module._validate_interface_quarantine_lineage(problem, audit)
    corrupted = deepcopy(audit)
    corrupted["statement_assessment"] = {"status": "quarantine", "issues": []}
    with pytest.raises(ValueError, match="interface_quarantine_statement_reclassified"):
        verifier_module._validate_interface_quarantine_lineage(
            problem, corrupted, original_statement_assessment
        )

    corrupted = deepcopy(audit)
    corrupted["retained_test_hashes"]["0"] = "0" * 64
    with pytest.raises(ValueError, match="interface_quarantine_test_hashes"):
        verifier_module._validate_interface_quarantine_lineage(problem, corrupted)


def test_interface_quarantine_is_visible_in_status_and_disposition_metrics():
    old = {
        "problem_id": "interactive-p1",
        "removed_indices": [],
        "disposition": "pending_domain_verification",
    }
    new = {
        "problem_id": "interactive-p1",
        "removed_indices": [],
        "disposition": "statement_quarantine",
        "direct_review_status": "quarantine_unsupported_interface",
        "direct_review_gate": "quarantine_unsupported_interface",
    }
    metrics = _recompute_direct_metrics(
        [old], [new],
        {"interactive-p1": {
            "status": "quarantine_unsupported_interface",
            "input_fail_indices": [],
            "output_fail_on_legal_indices": [],
        }},
        provisional_ids=set(),
    )
    assert metrics["new_direct_test_pairs_removed"] == 0
    assert metrics["direct_review_status_counts"] == {"quarantine_unsupported_interface": 1}
    assert metrics["direct_review_gate_counts"] == {"quarantine_unsupported_interface": 1}
    assert metrics["dispositions"] == {"statement_quarantine": 1}
    assert metrics["provisional_direct_candidates"] == 0


def test_interface_quarantine_gate_note_cannot_be_mislabeled_as_statement_corruption():
    note = merge_module.INTERFACE_QUARANTINE_GATE_NOTE
    assert note == verifier_module.INTERFACE_QUARANTINE_GATE_NOTE
    assert "statement is not classified as corrupt" in note
    report = {"direct_review_gate_notes": {"quarantine_unsupported_interface": note}}
    status_counts = {"quarantine_unsupported_interface": 1}

    verifier_module._validate_interface_quarantine_gate_note(report, status_counts)
    corrupted = deepcopy(report)
    corrupted["direct_review_gate_notes"]["quarantine_unsupported_interface"] = (
        "Whole problem is quarantined because the problem statement is missing."
    )
    with pytest.raises(ValueError, match="interface_quarantine_gate_note_mismatch"):
        verifier_module._validate_interface_quarantine_gate_note(corrupted, status_counts)


def test_test_pair_binding_hold_is_visible_without_masking_other_source_defects():
    old = {
        "problem_id": "pair-binding-p1",
        "removed_indices": [],
        "disposition": "pending_domain_verification",
    }
    new = {
        "problem_id": "pair-binding-p1",
        "removed_indices": [],
        "disposition": "pending_domain_verification",
        "direct_review_status": "hold_corrupt_test_binding",
        "direct_review_gate": "hold_corrupt_test_binding",
    }
    decision = {
        "status": "hold_corrupt_test_binding",
        "input_fail_indices": [],
        "output_fail_on_legal_indices": [1],
    }
    metrics = _recompute_direct_metrics(
        [old], [new], {"pair-binding-p1": decision}, provisional_ids=set()
    )
    assert metrics["new_direct_test_pairs_removed"] == 0
    assert metrics["direct_review_status_counts"] == {"hold_corrupt_test_binding": 1}
    assert metrics["direct_review_gate_counts"] == {"hold_corrupt_test_binding": 1}
    assert metrics["dispositions"] == {"pending_domain_verification": 1}
    assert metrics["provisional_direct_candidates"] == 0

    note = merge_module.TEST_PAIR_BINDING_GATE_NOTE
    assert note == verifier_module.TEST_PAIR_BINDING_GATE_NOTE
    assert "may coexist with statement or constraint defects" in note
    report = {"direct_review_gate_notes": {"hold_corrupt_test_binding": note}}
    status_counts = {"hold_corrupt_test_binding": 1}
    verifier_module._validate_test_pair_binding_gate_note(report, status_counts)
    report["direct_review_gate_notes"]["hold_corrupt_test_binding"] = (
        "The statement is corrupt and needs repair."
    )
    with pytest.raises(ValueError, match="test_pair_binding_gate_note_mismatch"):
        verifier_module._validate_test_pair_binding_gate_note(report, status_counts)


def test_nonunique_output_checker_quarantine_keeps_statement_and_test_lineage(problem):
    status = "quarantine_unsupported_checker"
    assert status in build_direct_review_decisions.DIRECT_OVERRIDE_STATUSES
    assert _expected_direct_gate({"disposition": "pending_domain_verification"},
                                 {"status": status}, False) == status

    statement_assessment = {"status": "no_confirmed_issue", "issues": []}
    audit = {"problem_id": problem.problem_id, "disposition": "statement_quarantine",
             "statement_assessment": deepcopy(statement_assessment)}
    merge_module._checker_quarantine_lineage(problem, audit)
    assert audit["statement_assessment"] == statement_assessment
    assert audit["checker_assessment"] == merge_module.CHECKER_QUARANTINE_ASSESSMENT
    assert audit["retained_indices"] == list(range(len(problem.tests)))
    assert audit["removed_indices"] == []
    verifier_module._validate_checker_quarantine_lineage(problem, audit, statement_assessment)

    corrupted = deepcopy(audit)
    corrupted["checker_assessment"]["reason"] = "missing_problem_statement"
    with pytest.raises(ValueError, match="checker_quarantine_assessment_mismatch"):
        verifier_module._validate_checker_quarantine_lineage(problem, corrupted, statement_assessment)

    note = merge_module.CHECKER_QUARANTINE_GATE_NOTE
    assert note == verifier_module.CHECKER_QUARANTINE_GATE_NOTE
    report = {"direct_review_gate_notes": {status: note}}
    verifier_module._validate_checker_quarantine_gate_note(report, {status: 1})


def test_provisional_gate_respects_base_blockers_and_minimum_test_count(problem):
    baseline = {
        "disposition": "pending_domain_verification",
        "retained_indices": list(range(len(problem.tests))),
        "statement_assessment": {"status": "no_confirmed_issue", "issues": []},
        "checker_review_flags": [{"reason": "checker review needed"}],
    }
    provisional = {
        "problem_id": problem.problem_id,
        "status": "provisional_repair_pending_coverage_and_fresh_backtest",
        "eligible_for_provisional_test_pruning": True,
        "unresolved_indices": [],
        "pair_pass_indices": [0, 1, 2, 3, 4],
    }
    assert _expected_direct_gate(baseline, provisional, False) == "hold_baseline_quality_blocker"

    fewer = {
        **provisional,
        "status": "quarantine_fewer_than_5_verified_tests",
        "eligible_for_provisional_test_pruning": False,
        "pair_pass_indices": [0, 1],
        "unresolved_indices": [2],
    }
    assert _expected_direct_gate(baseline, fewer, False) == "ineligible_fewer_than_5_verified_tests"


def test_statement_quarantine_must_keep_and_hash_every_original_test(problem):
    audit = {
        "disposition": "statement_quarantine",
        "removed_indices": [],
        "removed_test_hashes": {},
        "retained_indices": list(range(len(problem.tests))),
        "retained_test_hashes": {
            str(i): digest(asdict(test)) for i, test in enumerate(problem.tests)
        },
        "retained_test_count": len(problem.tests),
        "removed_test_count": 0,
    }
    _validate_statement_quarantine_lineage(problem, audit)

    corrupt = deepcopy(audit)
    corrupt["retained_indices"].pop()
    with pytest.raises(ValueError, match="statement_quarantine_retained_indices"):
        _validate_statement_quarantine_lineage(problem, corrupt)

    corrupt = deepcopy(audit)
    corrupt["retained_test_hashes"]["0"] = "0" * 64
    with pytest.raises(ValueError, match="statement_quarantine_test_hashes"):
        _validate_statement_quarantine_lineage(problem, corrupt)


def test_historical_subset_proof_is_recomputed_and_compared(problem):
    record = {
        "problem_id": problem.problem_id,
        "complete": True,
        "audit": [{
            "reference_index": 0,
            "result": {
                "tests_hash": problem.tests_hash,
                "code_hash": digest(problem.references[0]),
                "valid": True,
                "evidence_level": "isolated_execution",
                "total": len(problem.tests),
                "cases": [
                    {"status": "ok", "exit_code": 0, "truncated": False, "stdout": test.stdout}
                    for test in problem.tests
                ],
            },
        }],
    }
    record["audit"][0]["result"]["cases"][1]["stdout"] = "wrong\n"
    retained = [0, 2, 3, 4]
    audit = {"retained_indices": retained,
             "historical_reference_evidence": historical_subset_evidence(problem, retained, record)}
    assert _validate_historical_subset_proof(problem, audit, record) == audit[
        "historical_reference_evidence"
    ]

    corrupt = deepcopy(audit)
    corrupt["historical_reference_evidence"]["retained_all_pass"] = False
    with pytest.raises(ValueError, match="historical_subset_proof_mismatch"):
        _validate_historical_subset_proof(problem, corrupt, record)


def test_direct_report_metrics_count_effective_removals_and_hold_statuses():
    old = {"problem_id": "p1", "removed_indices": [1], "disposition": "pending_domain_verification"}
    new = {
        "problem_id": "p1",
        "removed_indices": [1, 4],
        "disposition": "direct_review_provisional_pending_fresh_backtest",
        "statement_assessment": {"status": "no_confirmed_issue", "issues": []},
        "checker_review_flags": [],
        "direct_review_status": "provisional_repair_pending_coverage_and_fresh_backtest",
        "direct_review_gate": "provisional_repair_pending_coverage_and_fresh_backtest",
    }
    hold_old = {"problem_id": "p2", "removed_indices": [], "disposition": "pending_domain_verification"}
    hold_new = {
        "problem_id": "p2",
        "removed_indices": [],
        "disposition": "pending_domain_verification",
        "direct_review_status": "hold_unresolved_test",
        "direct_review_gate": "hold_unresolved_test",
    }
    decisions = {
        "p1": {
            "status": "provisional_repair_pending_coverage_and_fresh_backtest",
            "input_fail_indices": [4],
            "output_fail_on_legal_indices": [5],
        },
        "p2": {
            "status": "hold_unresolved_test",
            "input_fail_indices": [],
            "output_fail_on_legal_indices": [],
        }
    }
    metrics = _recompute_direct_metrics(
        [old, hold_old], [new, hold_new], decisions, provisional_ids={"p1"})
    assert metrics["new_direct_test_pairs_removed"] == 1
    assert metrics["new_direct_illegal_input_pairs_removed"] == 1
    assert metrics["new_direct_bad_expected_output_pairs_removed"] == 0
    assert metrics["direct_review_status_counts"] == {
        "provisional_repair_pending_coverage_and_fresh_backtest": 1,
        "hold_unresolved_test": 1,
    }
    assert metrics["direct_review_gate_counts"] == {
        "provisional_repair_pending_coverage_and_fresh_backtest": 1,
        "hold_unresolved_test": 1,
    }
    assert metrics["dispositions"] == {
        "direct_review_provisional_pending_fresh_backtest": 1,
        "pending_domain_verification": 1,
    }
    assert metrics["provisional_direct_candidates"] == 1
