import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reports" / "backtest-20260928" / "verify_eval_backup.py"
SPEC = importlib.util.spec_from_file_location("verify_eval_backup", SCRIPT)
verify_eval_backup = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(verify_eval_backup)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixture_run(tmp_path):
    task = {
        "benchmark": "HumanEval+",
        "task_id": "HumanEval/0",
        "row_sha256": _sha("frozen-row"),
        "derived_prompt": "def f(x):\n",
        "row": {
            "task_id": "HumanEval/0",
            "canonical_solution": "def f(x): return x",
            "plus_input": [[1], [2], [3]],
            "test": "assert f(1) == 1",
        },
        "plus_coverage": "scoreable",
    }
    task_index = [{
        "benchmark": task["benchmark"],
        "task_id": task["task_id"],
        "source_row_sha256": task["row_sha256"],
        "plus_inputs": 3,
        "plus_coverage": "scoreable",
    }]
    manifest = {
        "selected_tasks": 1,
        "task_index": task_index,
        "identity": {
            "selection_mode": "explicit_task_ids",
            "selection_sha256": verify_eval_backup.digest(task_index),
            "selected_task_ids_sha256": verify_eval_backup.digest([task["task_id"]]),
            "chunk_size": 2,
        },
    }
    run = tmp_path / "evalplus"
    task_dir = run / "tasks" / "humaneval" / "HumanEval-0"
    intent_dir = run / "intents"
    task_dir.mkdir(parents=True)
    intent_dir.mkdir()
    record = {
        "benchmark": task["benchmark"],
        "task_id": task["task_id"],
        "attempt": 1,
        "source_row_sha256": task["row_sha256"],
        "canonical_source_sha256": verify_eval_backup.digest(
            task["derived_prompt"] + task["row"]["canonical_solution"]
        ),
        "base_assertion_sha256": verify_eval_backup.digest("assert f(1) == 1"),
        "base_executor_result": {
            "status": "ok", "physical_executions": 1, "elapsed_s": 0.1,
            "exit_code": 0, "truncated": False, "stdout_sha256": _sha("base"),
            "detail": "",
        },
        # A base failure is a valid recorded outcome, not an integrity error.
        "base_assertion_state": "failed",
        "base_assertion_processed": 1,
        "base_assertion_incomplete": 0,
        "plus_coverage": "scoreable",
        "plus_inputs_planned": 3,
        "plus_chunks": [
            {
                "indices": [0, 1],
                "executor_result": {
                    "status": "ok", "physical_executions": 1, "elapsed_s": 0.1,
                    "exit_code": 0, "truncated": False, "stdout_sha256": _sha("plus-0"),
                    "detail": "",
                },
                "state": "complete", "processed": 2, "incomplete": 0,
                "serialization_ok": 2, "serialization_failed": 0,
                "runtime_errors": 0, "policy_matches": 0,
                "policy_conflicts": 0, "unscored": 0,
            },
            {
                "indices": [2],
                "executor_result": {
                    "status": "ok", "physical_executions": 1, "elapsed_s": 0.1,
                    "exit_code": 0, "truncated": False, "stdout_sha256": _sha("plus-1"),
                    "detail": "",
                },
                "state": "complete", "processed": 1, "incomplete": 0,
                "serialization_ok": 0, "serialization_failed": 1,
                "runtime_errors": 0, "policy_matches": 0,
                "policy_conflicts": 0, "unscored": 0,
            },
        ],
        "plus_execution": {
            "planned": 3, "processed": 3, "incomplete": 0,
            "serialization_ok": 2, "serialization_failed": 1,
            "runtime_errors": 0, "policy_matches": 0,
            "policy_conflicts": 0, "policy_unscored_planned": 0,
            "unscored": 0,
        },
        "plus_execution_complete": True,
        "plus_correctness_evidence": "execution_and_serialization_only",
    }
    record_path = task_dir / "attempt-0001.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    intent = {
        "benchmark": task["benchmark"], "task_id": task["task_id"],
        "attempt": 1, "status": "dispatched", "planned_plus_inputs": 3,
        "source_row_sha256": task["row_sha256"],
        "task_record_path": "/remote/evalplus/tasks/humaneval/HumanEval-0/attempt-0001.json",
    }
    (intent_dir / "0000-HumanEval-0.json").write_text(json.dumps(intent), encoding="utf-8")
    status = {
        "selected_tasks": 1,
        "task_records_written": 1,
        "unfinished_or_unknown_tasks": 0,
        "base_assertions_passed": 0,
        "base_assertions_failed": 1,
        "base_assertions_unknown": 0,
        "plus_inputs_planned": 3,
        "plus_inputs_processed": 3,
        "plus_inputs_incomplete": 0,
        "serialization_ok": 2,
        "serialization_failed": 1,
        "runtime_errors": 0,
        "policy_matches": 0,
        "policy_conflicts": 0,
        "policy_unscored_cases_planned": 0,
        "unscored_cases_processed": 0,
    }
    return run, manifest, [task], status


def test_validates_self_consistent_failure_without_calling_it_a_pass(tmp_path):
    run, manifest, plan_tasks, status = _fixture_run(tmp_path)

    result = verify_eval_backup._validate_task_records(
        run, manifest, plan_tasks, status, evidence_hashes={}
    )

    assert result["summary"]["base_assertions_failed"] == 1
    assert result["problem_task_ids"] == ["HumanEval/0"]
    assert result["base_pass_task_ids"] == []


def test_rejects_gap_in_plus_case_indices(tmp_path):
    run, manifest, plan_tasks, status = _fixture_run(tmp_path)
    record_path = run / "tasks" / "humaneval" / "HumanEval-0" / "attempt-0001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["plus_chunks"][0]["indices"] = [0, 2]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="plus_chunk_index_mismatch"):
        verify_eval_backup._validate_task_records(
            run, manifest, plan_tasks, status, evidence_hashes={}
        )


def test_keeps_executor_timeout_incomplete_instead_of_rejecting_its_record(tmp_path):
    run, manifest, plan_tasks, status = _fixture_run(tmp_path)
    record_path = run / "tasks" / "humaneval" / "HumanEval-0" / "attempt-0001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    timeout_chunk = record["plus_chunks"][1]
    timeout_chunk["executor_result"].update(
        status="program_timeout", physical_executions=1
    )
    timeout_chunk.update(state="timed_out", processed=0, incomplete=1)
    for name in (
        "serialization_ok", "serialization_failed", "runtime_errors",
        "policy_matches", "policy_conflicts", "unscored",
    ):
        timeout_chunk.pop(name)
    plus = record["plus_execution"]
    plus.update(processed=2, incomplete=1, serialization_ok=2, serialization_failed=0)
    record["plus_execution_complete"] = False
    record_path.write_text(json.dumps(record), encoding="utf-8")
    status.update(
        plus_inputs_processed=2, plus_inputs_incomplete=1,
        serialization_ok=2, serialization_failed=0,
        unfinished_or_unknown_tasks=1,
    )

    result = verify_eval_backup._validate_task_records(
        run, manifest, plan_tasks, status, evidence_hashes={}
    )

    assert result["summary"]["plus_inputs_incomplete"] == 1
    assert result["problem_reasons"]["HumanEval/0"] == ["base_failed", "plus_incomplete"]
