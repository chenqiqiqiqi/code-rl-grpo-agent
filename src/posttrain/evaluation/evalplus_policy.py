"""Pure, hash-bound dispositions for a project-derived EvalPlus revision.

The official EvalPlus assets and evaluator stay untouched. Callers must first
check the pinned asset hash, then pass each test case's index and canonical
input hash to :func:`apply_case_policy` before considering an override.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from posttrain.common import digest


def require_pinned_asset(expected_sha256: str, observed_sha256: str) -> None:
    """Refuse to apply this policy to a different frozen dataset asset."""
    if expected_sha256 != observed_sha256:
        raise ValueError("asset hash mismatch for EvalPlus remediation policy")


def task_disposition(policy: Mapping[str, Any], benchmark: str, task_id: str) -> str:
    """Return a task-level derived disposition, defaulting to official behavior."""
    task = policy.get("benchmarks", {}).get(benchmark, {}).get("tasks", {}).get(task_id)
    if task is None:
        return "retain_official"
    return str(task.get("derived_disposition", "retain_official"))


def plus_coverage_disposition(plus_input: Any) -> str:
    """Do not call an empty or malformed plus-test collection a pass."""
    if isinstance(plus_input, list):
        if not plus_input:
            return "unscored_empty_plus"
        if all(isinstance(args, (list, tuple)) for args in plus_input):
            return "scoreable"
        return "unscored_invalid_plus_container"
    if isinstance(plus_input, dict) and not plus_input:
        # MBPP v0.2.0 Mbpp/793 stores this exact empty object.
        return "unscored_empty_plus"
    return "unscored_invalid_plus_container"


def _single_argument(args: Sequence[Any], task_id: str) -> Any:
    if not isinstance(args, (list, tuple)) or len(args) != 1:
        raise ValueError(f"{task_id} remediation expects exactly one input argument")
    return args[0]


def _require_positive_integer(value: Any, task_id: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{task_id} remediation requires a positive integer input")
    return value


def _maximum_uppercase_run(text: str) -> int:
    longest = current = 0
    for character in text:
        if character.isupper():
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def apply_prompt_policy(
    task_policy: Mapping[str, Any], task_id: str, prompt: str
) -> str:
    """Apply exact prompt replacements only to the pinned original prompt."""
    if task_policy.get("task_id") != task_id:
        raise ValueError(
            f"task id mismatch: policy is for {task_policy.get('task_id')!r}, "
            f"received {task_id!r}"
        )
    if not isinstance(prompt, str):
        raise ValueError(f"{task_id} prompt remediation expects a string prompt")
    if task_policy.get("source_prompt_sha256") != digest(prompt):
        raise ValueError(f"source prompt hash mismatch for {task_id}")

    patches = task_policy.get("prompt_patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError(f"no prompt patches registered for {task_id}")

    derived_prompt = prompt
    for patch in patches:
        old_text = patch.get("old_text")
        new_text = patch.get("new_text")
        if not isinstance(old_text, str) or not old_text:
            raise ValueError(f"invalid prompt patch anchor for {task_id}")
        if not isinstance(new_text, str):
            raise ValueError(f"invalid prompt patch replacement for {task_id}")
        if derived_prompt.count(old_text) != 1:
            raise ValueError(
                f"prompt patch anchor for {task_id} must match exactly once"
            )
        derived_prompt = derived_prompt.replace(old_text, new_text, 1)

    if task_policy.get("derived_prompt_sha256") != digest(derived_prompt):
        raise ValueError(f"derived prompt hash mismatch for {task_id}")
    return derived_prompt


def apply_case_policy(
    task_policy: Mapping[str, Any],
    task_id: str,
    plus_input_index: int,
    args: Sequence[Any],
    input_sha256: str,
) -> dict[str, Any]:
    """Apply one registered case rule after checking its index and input hash.

    Unlisted cases retain the official expected output. Registered overrides
    are limited to the audited plus-input indices in ``policy.json``.
    """
    if task_policy.get("task_id") != task_id:
        raise ValueError(
            f"task id mismatch: policy is for {task_policy.get('task_id')!r}, "
            f"received {task_id!r}"
        )

    matching_rules = [
        rule
        for rule in task_policy.get("case_rules", [])
        if rule.get("index") == plus_input_index
    ]
    if len(matching_rules) > 1:
        raise ValueError(
            f"duplicate case rule for {task_id} plus case {plus_input_index}"
        )
    case_rule = matching_rules[0] if matching_rules else None

    if case_rule is None:
        return {"disposition": "retain_official", "expected": None}

    actual_sha256 = digest(args)
    if input_sha256 != actual_sha256 or case_rule.get("input_sha256") != actual_sha256:
        raise ValueError(
            f"input hash mismatch for {task_id} plus case {plus_input_index}"
        )

    action = case_rule.get("action")
    if action == "unscoreable_outside_integer_contract" and task_id == "Mbpp/267":
        return {"disposition": "unscored", "expected": None}

    if action == "exact_odd_square_sum" and task_id == "Mbpp/267":
        n = _require_positive_integer(_single_argument(args, task_id), task_id)
        return {
            "disposition": "replace_expected",
            "expected": n * (4 * n * n - 1) // 3,
        }

    if action == "exact_even_square_sum" and task_id == "Mbpp/287":
        n = _require_positive_integer(_single_argument(args, task_id), task_id)
        return {
            "disposition": "replace_expected",
            "expected": 2 * n * (n + 1) * (2 * n + 1) // 3,
        }

    if action == "maximum_uppercase_run" and task_id == "Mbpp/806":
        text = _single_argument(args, task_id)
        if not isinstance(text, str):
            raise ValueError(f"{task_id} remediation requires a string input")
        return {
            "disposition": "replace_expected",
            "expected": _maximum_uppercase_run(text),
        }

    raise ValueError(f"unsupported remediation action for {task_id}: {action!r}")
