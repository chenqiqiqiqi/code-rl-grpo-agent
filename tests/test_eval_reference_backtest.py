from __future__ import annotations

import json
import ast
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

from posttrain.execution.contracts import CaseResult, Status
from posttrain.evaluation.reference_backtest import (
    MARKER_PREFIX,
    build_backtest_plan,
    chunk_plus_inputs,
    interpret_worker_result,
    prepare_task_run,
    run_reference_backtest,
    normalize_official_output,
    _worker_source,
)
from posttrain.common import digest


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_plan_selects_540_evalplus_tasks_and_leaves_lcb_without_references():
    plan = build_backtest_plan(ROOT)

    assert len(plan["runnable_tasks"]) == 540
    assert {key: plan["inventory"]["LCB-stdio-200"][key] for key in (
        "manifest_task_count", "held_task_ids", "reference_backtestable_count", "blocked_no_reference_count"
    )} == {
        "manifest_task_count": 200,
        "held_task_ids": ["atcoder:arc184_c"],
        "reference_backtestable_count": 0,
        "blocked_no_reference_count": 199,
    }
    assert plan["inventory"]["EvalPlus"]["held_task_ids"] == [
        "HumanEval/116",
        "HumanEval/68",
    ]
    assert plan["inventory"]["EvalPlus"]["runnable_plus_input_count"] == 160802
    assert plan["inventory"]["EvalPlus"]["default_plus_chunk_count"] == 1389
    assert plan["inventory"]["EvalPlus"]["default_worker_child_invocation_count"] == 1929
    assert plan["inventory"]["EvalPlus"]["empty_plus_task_ids"] == ["Mbpp/793"]


def test_mbpp_793_keeps_base_and_marks_empty_plus_unscored():
    plan = build_backtest_plan(ROOT)
    task = next(t for t in plan["runnable_tasks"] if t["task_id"] == "Mbpp/793")

    prepared = prepare_task_run(task, chunk_size=8)

    assert prepared["plus_coverage"] == "unscored_empty_plus"
    assert prepared["plus_input_count"] == 0
    assert len(prepared["stdins"]) == 1  # independent original base assertions
    assert json.loads(prepared["stdins"][0])["kind"] == "base"


def test_plus_chunks_preserve_case_indices_and_do_not_collapse_inputs():
    cases = [[i] for i in range(11)]

    chunks = chunk_plus_inputs(cases, chunk_size=4)

    assert [[case["index"] for case in chunk] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10],
    ]
    assert [case["args"] for chunk in chunks for case in chunk] == cases


def test_timeout_chunk_is_unknown_and_never_counted_as_completed():
    interpreted = interpret_worker_result(
        CaseResult(Status.TIMEOUT, physical_executions=1),
        expected_kind="plus",
        planned_cases=7,
    )

    assert interpreted["state"] == "timed_out"
    assert interpreted["processed"] == 0
    assert interpreted["incomplete"] == 7


def test_worker_marker_must_match_expected_kind_and_case_count():
    wrong_kind = CaseResult(Status.OK, stdout=MARKER_PREFIX + '{"kind":"base","status":"passed"}')
    malformed = interpret_worker_result(wrong_kind, expected_kind="plus", planned_cases=3)

    assert malformed["state"] == "incomplete"
    assert malformed["processed"] == 0
    assert malformed["incomplete"] == 3


def test_worker_marker_with_self_contradictory_counts_is_unknown():
    marker = {
        "kind": "plus",
        "planned": 3,
        "processed": 3,
        "serialization_ok": 3,
        "serialization_failed": 0,
        "runtime_errors": 1,
        "official_special_normalized": 0,
        "normalized_serialization_ok": 0,
        "normalized_serialization_failed": 0,
        "policy_matches": 0,
        "policy_conflicts": 0,
        "unscored": 0,
    }
    interpreted = interpret_worker_result(
        CaseResult(Status.OK, stdout=MARKER_PREFIX + json.dumps(marker)),
        expected_kind="plus",
        planned_cases=3,
    )

    assert interpreted["state"] == "unknown"
    assert interpreted["processed"] == 0
    assert interpreted["incomplete"] == 3


def test_policy_match_and_unscored_counters_cannot_claim_the_same_case_twice():
    marker = {
        "kind": "plus",
        "planned": 1,
        "processed": 1,
        "serialization_ok": 1,
        "serialization_failed": 0,
        "runtime_errors": 0,
        "official_special_normalized": 0,
        "normalized_serialization_ok": 0,
        "normalized_serialization_failed": 0,
        "policy_matches": 1,
        "policy_conflicts": 0,
        "unscored": 1,
    }

    interpreted = interpret_worker_result(
        CaseResult(Status.OK, stdout=MARKER_PREFIX + json.dumps(marker)),
        expected_kind="plus",
        planned_cases=1,
    )

    assert interpreted["state"] == "unknown"
    assert interpreted["processed"] == 0


def test_actual_mbpp_287_policy_case_is_bound_into_plus_payload():
    plan = build_backtest_plan(ROOT)
    task = next(t for t in plan["runnable_tasks"] if t["task_id"] == "Mbpp/287")
    rule = task["task_policy"]["case_rules"][0]

    prepared = prepare_task_run(task, chunk_size=128)

    payloads = [json.loads(value) for value in prepared["stdins"][1:]]
    case = next(case for payload in payloads for case in payload["cases"] if case["index"] == rule["index"])
    n = case["args"][0]
    assert case["policy"] == {"disposition": "replace_expected", "expected": 2 * n * (n + 1) * (2 * n + 1) // 3}


def test_actual_mbpp_267_keeps_three_noninteger_cases_unscored():
    plan = build_backtest_plan(ROOT)
    task = next(t for t in plan["runnable_tasks"] if t["task_id"] == "Mbpp/267")

    prepared = prepare_task_run(task, chunk_size=128)

    payloads = [json.loads(value) for value in prepared["stdins"][1:]]
    cases = [case for payload in payloads for case in payload["cases"]]
    unscored = [case for case in cases if case["policy"]["disposition"] == "unscored"]
    assert len(unscored) == 3
    assert all("expected" not in case["policy"] for case in unscored)


def test_actual_he_47_prompt_uses_hash_bound_derived_example():
    plan = build_backtest_plan(ROOT)
    task = next(t for t in plan["runnable_tasks"] if t["task_id"] == "HumanEval/47")

    assert task["derived_prompt"] != task["row"]["prompt"]
    assert "    8.0" in task["derived_prompt"]
    assert "    15.0" not in task["derived_prompt"]


def test_mbpp_120_base_wrapper_binds_math_without_changing_raw_assertions():
    plan = build_backtest_plan(ROOT)
    task = next(t for t in plan["runnable_tasks"] if t["task_id"] == "Mbpp/120")

    prepared = prepare_task_run(task, chunk_size=128)

    assert "namespace[\"math\"] = math" in prepared["code"]
    assert "math.isclose" in task["row"]["assertion"]
    assert "import math" not in task["row"]["prompt"]
    assert "import math" not in task["row"]["canonical_solution"]


def test_synthetic_mbpp_base_assertion_can_resolve_math_without_source_import():
    task = {
        "benchmark": "MBPP+",
        "task_id": "Mbpp/120",
        "derived_prompt": "",
        "row": {
            "entry_point": "sample",
            "canonical_solution": "def sample():\n    return 0.5\n",
            "assertion": "assert math.isclose(sample(), 0.5)",
        },
    }
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _worker_source(task)],
        input=json.dumps({"kind": "base"}),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    marker = json.loads(completed.stdout.splitlines()[-1][len(MARKER_PREFIX):])
    assert marker == {"kind": "base", "status": "passed"}


def test_official_mbpp_not_none_oracle_normalizes_unserializable_match_outputs():
    raw = re.search("x", "x")
    with pytest.raises(TypeError):
        pickle.dumps(raw)

    normalized = normalize_official_output("MBPP+", "check_str", raw)

    assert normalized is True
    assert type(normalized) is bool
    assert pickle.dumps(normalized)
    assert normalize_official_output("MBPP+", "check_str", None) is False
    assert normalize_official_output("MBPP+", "check_str", False) is False
    assert normalize_official_output("MBPP+", "max_product_tuple", raw) is raw


def test_synthetic_worker_keeps_raw_pickle_failure_and_records_official_normalization():
    task = {
        "benchmark": "MBPP+",
        "task_id": "Mbpp/737",
        "derived_prompt": "",
        "row": {
            "entry_point": "check_str",
            "canonical_solution": "import re\ndef check_str(pattern, text):\n    return re.search(pattern, text)\n",
            "assertion": "assert check_str('x', 'x') is not None",
        },
    }
    payload = {
        "kind": "plus",
        "cases": [{"index": 0, "args": ["x", "x"], "policy": {"disposition": "none"}}],
    }
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _worker_source(task)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    marker = json.loads(completed.stdout.splitlines()[-1][len(MARKER_PREFIX):])
    assert marker["serialization_failed"] == 1
    assert marker["official_special_normalized"] == 1
    assert marker["normalized_serialization_ok"] == 1
    assert marker["normalized_serialization_failed"] == 0
    interpreted = interpret_worker_result(
        CaseResult(Status.OK, stdout=MARKER_PREFIX + json.dumps(marker)),
        expected_kind="plus",
        planned_cases=1,
    )
    assert interpreted["state"] == "complete"
    assert interpreted["serialization_failed"] == 1
    assert interpreted["normalized_serialization_ok"] == 1


def test_737_and_794_use_the_pinned_official_not_none_task_names():
    plan = build_backtest_plan(ROOT)
    by_id = {t["task_id"]: t for t in plan["runnable_tasks"]}

    assert by_id["Mbpp/737"]["row"]["entry_point"] == "check_str"
    assert by_id["Mbpp/794"]["row"]["entry_point"] == "text_starta_endb"
    for task_id in ("Mbpp/737", "Mbpp/794"):
        prepared = prepare_task_run(by_id[task_id], chunk_size=128)
        assert "official_special_normalized" in prepared["code"]


def test_all_540_tasks_can_be_prepared_without_executing_reference_code():
    plan = build_backtest_plan(ROOT)

    prepared = [prepare_task_run(task, chunk_size=128) for task in plan["runnable_tasks"]]

    assert len(prepared) == 540
    assert all(1 <= len(item["stdins"]) <= 4096 for item in prepared)
    assert any("Infinity" in stdin for item in prepared for stdin in item["stdins"][1:])
    assert all(ast.parse(item["code"]) for item in prepared)


class _MarkerBackend:
    evidence_level = "isolated_execution"
    version = "synthetic-no-execution-backend-v1"

    def run_batch(self, code, stdins, limits, request_id, timeout_s):
        assert code  # source was prepared, not executed in this test
        responses = []
        for stdin in stdins:
            payload = json.loads(stdin)
            if payload["kind"] == "base":
                marker = {"kind": "base", "status": "passed"}
            else:
                marker = {
                    "kind": "plus",
                    "planned": len(payload["cases"]),
                    "processed": len(payload["cases"]),
                    "serialization_ok": len(payload["cases"]),
                    "serialization_failed": 0,
                    "runtime_errors": 0,
                    "official_special_normalized": 0,
                    "normalized_serialization_ok": 0,
                    "normalized_serialization_failed": 0,
                    "policy_matches": 0,
                    "policy_conflicts": 0,
                    "unscored": sum(c["policy"]["disposition"] == "unscored" for c in payload["cases"]),
                }
            responses.append(CaseResult(Status.OK, stdout=MARKER_PREFIX + json.dumps(marker)))
        return responses


def test_run_persists_per_task_attempt_and_never_calls_plus_a_correctness_pass(tmp_path):
    output = tmp_path / "run"
    result = run_reference_backtest(
        ROOT,
        output,
        _MarkerBackend(),
        workers=1,
        max_wall_hours=0.1,
        task_timeout_seconds=30,
        chunk_size=8,
        max_tasks=1,
    )

    assert result["selected_tasks"] == 1
    record_paths = list((output / "tasks").rglob("attempt-0001.json"))
    assert len(record_paths) == 1
    record = json.loads(record_paths[0].read_text(encoding="utf-8"))
    assert record["base_assertion_state"] == "passed"
    assert record["plus_correctness_evidence"] == "execution_and_serialization_only"
    assert record["plus_execution_complete"] is True
    assert record["base_executor_result"]["physical_executions"] == 1
    assert record["plus_chunks"][0]["executor_result"]["status"] == "ok"


def test_task_id_subset_records_source_plan_and_selection_hash(tmp_path):
    selected = ["Mbpp/120", "Mbpp/737", "Mbpp/794"]
    output = tmp_path / "subset"

    result = run_reference_backtest(
        ROOT,
        output,
        _MarkerBackend(),
        workers=1,
        max_wall_hours=0.1,
        task_timeout_seconds=30,
        chunk_size=128,
        task_ids=selected,
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["source_plan_tasks"] == 540
    assert result["selected_tasks"] == 3
    assert manifest["identity"]["source_plan_task_count"] == 540
    assert manifest["identity"]["selected_task_ids_sha256"] == digest(selected)
    assert manifest["identity"]["selection_mode"] == "explicit_task_ids"
    assert [row["task_id"] for row in manifest["task_index"]] == selected
    assert result["selected_task_ids"] == selected


@pytest.mark.parametrize(
    ("task_ids", "message"),
    [
        (["Mbpp/120", "Mbpp/120"], "duplicate_task_ids"),
        (["Unknown/1"], "unknown_task_ids"),
        (["HumanEval/68"], "held_task_ids"),
    ],
)
def test_invalid_task_id_subsets_are_rejected_before_output_creation(tmp_path, task_ids, message):
    output = tmp_path / "invalid"

    with pytest.raises(ValueError, match=message):
        run_reference_backtest(
            ROOT,
            output,
            _MarkerBackend(),
            workers=1,
            max_wall_hours=0.1,
            task_timeout_seconds=30,
            task_ids=task_ids,
        )
    assert not output.exists()


def test_existing_output_refuses_duplicate_dispatch(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refuse_duplicate_dispatch"):
        run_reference_backtest(
            ROOT,
            output,
            _MarkerBackend(),
            workers=1,
            max_wall_hours=0.1,
            task_timeout_seconds=30,
            chunk_size=8,
            max_tasks=1,
        )
