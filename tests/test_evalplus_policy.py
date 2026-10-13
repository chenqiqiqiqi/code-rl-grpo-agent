"""Synthetic tests for hash-bound, project-derived EvalPlus dispositions."""

from __future__ import annotations

import importlib

import pytest

from posttrain.common import digest


def _policy_api():
    try:
        return importlib.import_module("posttrain.evaluation.evalplus_policy")
    except ImportError:
        return None


def _task_policy(action: str, task_id: str, args: list) -> dict:
    return {
        "task_id": task_id,
        "case_rules": [
            {
                "index": 0,
                "input_sha256": digest(args),
                "action": action,
            }
        ]
    }


def test_integer_oracle_repair_uses_exact_odd_square_sum_for_hash_bound_case():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    decision = api.apply_case_policy(
        _task_policy("exact_odd_square_sum", "Mbpp/267", [2]),
        "Mbpp/267",
        0,
        [2],
        digest([2]),
    )

    assert decision == {"disposition": "replace_expected", "expected": 10}


def test_integer_oracle_repair_uses_exact_even_square_sum_for_hash_bound_case():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    decision = api.apply_case_policy(
        _task_policy("exact_even_square_sum", "Mbpp/287", [2]),
        "Mbpp/287",
        0,
        [2],
        digest([2]),
    )

    assert decision == {"disposition": "replace_expected", "expected": 20}


def test_uppercase_run_repair_returns_longest_run_not_last_run():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    decision = api.apply_case_policy(
        _task_policy("maximum_uppercase_run", "Mbpp/806", ["AaaBBcC"]),
        "Mbpp/806",
        0,
        ["AaaBBcC"],
        digest(["AaaBBcC"]),
    )

    assert decision == {"disposition": "replace_expected", "expected": 2}


def test_case_hash_mismatch_refuses_to_apply_a_derived_oracle():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    with pytest.raises(ValueError, match="input hash mismatch"):
        api.apply_case_policy(
            _task_policy("exact_odd_square_sum", "Mbpp/267", [2]),
            "Mbpp/267",
            0,
            [2],
            "other-hash",
        )


def test_old_matching_hash_cannot_be_reused_with_different_arguments():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    original_hash = digest([2])
    with pytest.raises(ValueError, match="input hash mismatch"):
        api.apply_case_policy(
            _task_policy("exact_odd_square_sum", "Mbpp/267", [2]),
            "Mbpp/267",
            0,
            [3],
            original_hash,
        )


def test_duplicate_case_indices_are_rejected_as_ambiguous_policy():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    policy = _task_policy("exact_odd_square_sum", "Mbpp/267", [2])
    policy["case_rules"].append(dict(policy["case_rules"][0]))

    with pytest.raises(ValueError, match="duplicate case rule"):
        api.apply_case_policy(policy, "Mbpp/267", 0, [2], digest([2]))


def test_policy_cannot_be_reused_for_a_different_task_id():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    with pytest.raises(ValueError, match="task id mismatch"):
        api.apply_case_policy(
            _task_policy("exact_odd_square_sum", "Mbpp/267", [2]),
            "Mbpp/287",
            0,
            [2],
            digest([2]),
        )


def test_noninteger_input_is_left_unscored_without_guessing_a_repair():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    decision = api.apply_case_policy(
        _task_policy("unscoreable_outside_integer_contract", "Mbpp/267", [2.5]),
        "Mbpp/267",
        0,
        [2.5],
        digest([2.5]),
    )

    assert decision == {"disposition": "unscored", "expected": None}


def test_out_of_contract_action_cannot_be_reused_for_another_task():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    with pytest.raises(ValueError, match="unsupported remediation action"):
        api.apply_case_policy(
            _task_policy("unscoreable_outside_integer_contract", "Mbpp/287", [2.5]),
            "Mbpp/287",
            0,
            [2.5],
            digest([2.5]),
        )


def test_empty_mbpp_plus_is_unscored_instead_of_a_vacuous_pass():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    assert api.plus_coverage_disposition({}) == "unscored_empty_plus"
    assert api.plus_coverage_disposition([]) == "unscored_empty_plus"
    assert api.plus_coverage_disposition([[1]]) == "scoreable"
    assert api.plus_coverage_disposition([[]]) == "scoreable"
    assert api.plus_coverage_disposition([None]) == "unscored_invalid_plus_container"


def test_unknown_humaneval_prompt_conflict_blocks_derived_task_result():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    policy = {
        "benchmarks": {
            "HumanEval+": {
                "tasks": {
                    "HumanEval/47": {
                        "derived_disposition": "blocked_prompt_conflict"
                    }
                }
            }
        }
    }

    assert (
        api.task_disposition(policy, "HumanEval+", "HumanEval/47")
        == "blocked_prompt_conflict"
    )
    assert (
        api.task_disposition(policy, "HumanEval+", "HumanEval/0")
        == "retain_official"
    )


def test_pinned_asset_check_rejects_a_different_release_file():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"

    with pytest.raises(ValueError, match="asset hash mismatch"):
        api.require_pinned_asset("expected-file-hash", "changed-file-hash")


def _prompt_policy(task_id: str, source_prompt: str, old_text: str, new_text: str) -> dict:
    derived_prompt = source_prompt.replace(old_text, new_text)
    return {
        "task_id": task_id,
        "source_prompt_sha256": digest(source_prompt),
        "derived_prompt_sha256": digest(derived_prompt),
        "prompt_patches": [{"old_text": old_text, "new_text": new_text}],
    }


def _apply_prompt_policy(api, task_policy: dict, task_id: str, prompt: str) -> str:
    apply = getattr(api, "apply_prompt_policy", None)
    assert callable(apply), "the hash-bound prompt policy is missing"
    return apply(task_policy, task_id, prompt)


def test_prompt_policy_corrects_median_example_only_for_bound_source_prompt():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "median([-10, 4, 6, 1000, 10, 20]) => 15.0"
    old_text = "median([-10, 4, 6, 1000, 10, 20]) => 15.0"
    new_text = "median([-10, 4, 6, 1000, 10, 20]) => 8.0"

    patched = _apply_prompt_policy(
        api,
        _prompt_policy("HumanEval/47", source, old_text, new_text),
        "HumanEval/47",
        source,
    )

    assert patched == "median([-10, 4, 6, 1000, 10, 20]) => 8.0"


def test_prompt_policy_corrects_sort_example_without_changing_other_examples():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]\nother example stays"
    old_text = "sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]"
    new_text = "sort_array([1, 5, 2, 3, 4]) == [1, 2, 4, 3, 5]"

    patched = _apply_prompt_policy(
        api,
        _prompt_policy("HumanEval/116", source, old_text, new_text),
        "HumanEval/116",
        source,
    )

    assert patched == (
        "sort_array([1, 5, 2, 3, 4]) == [1, 2, 4, 3, 5]\nother example stays"
    )


def test_prompt_policy_rejects_changed_prompt_even_when_patch_anchor_matches():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "median example: 15.0"
    old_text = "15.0"
    policy = _prompt_policy("HumanEval/47", source, old_text, "8.0")

    with pytest.raises(ValueError, match="source prompt hash mismatch"):
        _apply_prompt_policy(api, policy, "HumanEval/47", "changed median example: 15.0")


def test_prompt_policy_rejects_duplicate_exact_patch_anchor():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "median example: 15.0; repeated median example: 15.0"
    policy = _prompt_policy("HumanEval/47", source, "15.0", "8.0")

    with pytest.raises(ValueError, match="must match exactly once"):
        _apply_prompt_policy(api, policy, "HumanEval/47", source)


def test_prompt_policy_rejects_cross_task_reuse():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "median example: 15.0"
    policy = _prompt_policy("HumanEval/47", source, "15.0", "8.0")

    with pytest.raises(ValueError, match="task id mismatch"):
        _apply_prompt_policy(api, policy, "HumanEval/116", source)


def test_prompt_policy_checks_the_declared_derived_digest():
    api = _policy_api()
    assert api is not None, "the EvalPlus remediation policy is missing"
    source = "median example: 15.0"
    policy = _prompt_policy("HumanEval/47", source, "15.0", "8.0")
    policy["derived_prompt_sha256"] = "stale-derived-hash"

    with pytest.raises(ValueError, match="derived prompt hash mismatch"):
        _apply_prompt_policy(api, policy, "HumanEval/47", source)
