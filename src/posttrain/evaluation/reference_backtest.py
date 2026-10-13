"""Small, resumability-safe controller for isolated EvalPlus reference checks.

The controller reads frozen assets and sends code plus inputs to the existing
attested DockerBackend. It never evaluates canonical code or assertions on the
controller host. Plus inputs are only execution/serialization diagnostics,
except for the separately labeled, hash-bound project policy comparisons.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from posttrain.common import digest, file_digest, utc_now, write_json
from posttrain.evaluation.evalplus_policy import (
    apply_case_policy,
    apply_prompt_policy,
    plus_coverage_disposition,
)
from posttrain.execution.contracts import CaseResult, Limits, Status


PINNED_EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
PINNED_SPECIAL_ORACLE_BLOB_SHA256 = "12201dc717b226485e30960381437321c53f9af4"
PINNED_EVAL_INIT_BLOB_SHA256 = "ded48b87c4315664ccc25a48bb135195cfd2dcd2"
FROZEN_LCB_MANIFEST_SHA256 = "140acae21336805788b6a600a2b0a7a6f1c5d3c36df94b00bd2a3fae16565ba2"
MARKER_PREFIX = "__POSTTRAIN_REFERENCE_BACKTEST__="
MBPP_OUTPUT_NOT_NONE_TASKS = frozenset(
    {"check_str", "text_match_three", "text_starta_endb"}
)
HOLD_TASK_IDS = {"HumanEval/68", "HumanEval/116"}
LCB_HOLD_TASK_IDS = {"atcoder:arc184_c"}
DEFAULT_CHUNK_SIZE = 128
MAX_WORKERS = 4
MAX_WALL_HOURS = 8.0


def normalize_official_output(benchmark: str, entry_point: str, output: Any) -> Any:
    """Apply the pinned EvalPlus MBPP non-None special oracle normalization."""
    if benchmark == "MBPP+" and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
        return output if isinstance(output, bool) else output is not None
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_jsonl_gzip(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    with gzip.open(path, "rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                row = json.loads(raw_line)
            except (UnicodeError, ValueError) as exc:
                raise ValueError(f"invalid_evalplus_jsonl:{path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"invalid_evalplus_row:{path}:{line_number}")
            rows.append((row, hashlib.sha256(raw_line).hexdigest()))
    return rows


def _validate_manifest_rows(
    benchmark: str,
    dataset: Mapping[str, Any],
    root: Path,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = root / str(dataset["local_file"])
    manifest_path = root / str(dataset["manifest"])
    if file_digest(raw_path) != dataset["local_file_sha256"]:
        raise ValueError(f"evalplus_asset_hash_mismatch:{benchmark}")
    if file_digest(manifest_path) != dataset["manifest_file_sha256"]:
        raise ValueError(f"evalplus_manifest_file_hash_mismatch:{benchmark}")

    manifest = _read_json(manifest_path)
    if manifest.get("manifest_hash") != dataset["manifest_embedded_hash"]:
        raise ValueError(f"evalplus_manifest_hash_mismatch:{benchmark}")
    manifest_rows = manifest.get("tasks")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != dataset["task_count"]:
        raise ValueError(f"evalplus_manifest_count_mismatch:{benchmark}")
    expected: dict[str, dict[str, Any]] = {}
    for entry in manifest_rows:
        task_id = entry.get("task_id") if isinstance(entry, dict) else None
        if not isinstance(task_id, str) or task_id in expected:
            raise ValueError(f"invalid_or_duplicate_evalplus_manifest_task:{benchmark}")
        expected[task_id] = entry

    rows = _read_jsonl_gzip(raw_path)
    if len(rows) != dataset["task_count"]:
        raise ValueError(f"evalplus_asset_count_mismatch:{benchmark}")
    task_policy = policy.get("benchmarks", {}).get(benchmark, {}).get("tasks", {})
    if not isinstance(task_policy, dict):
        raise ValueError(f"invalid_evalplus_task_policy:{benchmark}")

    found: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for row, row_sha256 in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected or task_id in found:
            raise ValueError(f"unexpected_or_duplicate_evalplus_row:{benchmark}")
        found.add(task_id)
        entry = expected[task_id]
        if row.get("entry_point") != entry.get("entry_point") or digest(row.get("prompt")) != entry.get("prompt_hash"):
            raise ValueError(f"evalplus_row_manifest_mismatch:{task_id}")
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("canonical_solution"), str):
            raise ValueError(f"invalid_evalplus_reference_fields:{task_id}")
        base = row.get("test", row.get("assertion"))
        if isinstance(base, list):
            if not all(isinstance(part, str) for part in base):
                raise ValueError(f"invalid_evalplus_base_assertions:{task_id}")
        elif not isinstance(base, str):
            raise ValueError(f"invalid_evalplus_base_assertions:{task_id}")

        registered = task_policy.get(task_id, {"task_id": task_id, "case_rules": []})
        if registered.get("task_id") != task_id:
            raise ValueError(f"evalplus_policy_task_mismatch:{task_id}")
        registered_row_hash = registered.get("source_row_sha256")
        if registered_row_hash is not None and registered_row_hash != row_sha256:
            raise ValueError(f"evalplus_policy_row_hash_mismatch:{task_id}")
        coverage = plus_coverage_disposition(row.get("plus_input"))
        prompt = row["prompt"]
        if registered.get("prompt_patches"):
            prompt = apply_prompt_policy(registered, task_id, prompt)
        tasks.append(
            {
                "benchmark": benchmark,
                "task_id": task_id,
                "row": row,
                "row_sha256": row_sha256,
                "task_policy": registered,
                "derived_prompt": prompt,
                "plus_coverage": coverage,
            }
        )
    if found != set(expected):
        raise ValueError(f"evalplus_manifest_coverage_mismatch:{benchmark}")
    # Preserve the pinned manifest order, which is the task order used in reports.
    by_id = {task["task_id"]: task for task in tasks}
    return [by_id[entry["task_id"]] for entry in manifest_rows], manifest


def build_backtest_plan(repo_root: Path) -> dict[str, Any]:
    """Load and hash-check frozen manifests/assets; do not execute their code."""
    root = Path(repo_root)
    policy_path = root / "reports" / "evalset-remediation" / "evalplus" / "policy.json"
    policy = _read_json(policy_path)
    official = policy.get("official_baseline", {})
    datasets = official.get("datasets", {})
    if official.get("pinned_evalplus_source_commit") != PINNED_EVALPLUS_COMMIT:
        raise ValueError("unexpected_pinned_evalplus_source_commit")
    loaded: dict[str, list[dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    asset_hashes: dict[str, str] = {}
    for benchmark in ("HumanEval+", "MBPP+"):
        dataset = datasets.get(benchmark)
        if not isinstance(dataset, dict):
            raise ValueError(f"missing_evalplus_asset_identity:{benchmark}")
        tasks, manifest = _validate_manifest_rows(benchmark, dataset, root, policy)
        loaded[benchmark] = tasks
        manifest_hashes[benchmark] = file_digest(root / dataset["manifest"])
        asset_hashes[benchmark] = str(dataset["local_file_sha256"])

    lcb_path = root / "manifests" / "lcb_stdio_200_v2.json"
    lcb_manifest = _read_json(lcb_path)
    lcb_manifest_sha256 = file_digest(lcb_path)
    if lcb_manifest_sha256 != FROZEN_LCB_MANIFEST_SHA256:
        raise ValueError("frozen_lcb_manifest_hash_mismatch")
    lcb_tasks = lcb_manifest.get("tasks")
    if not isinstance(lcb_tasks, list) or len(lcb_tasks) != 200:
        raise ValueError("frozen_lcb_manifest_count_mismatch")
    lcb_ids = [row.get("task_id") for row in lcb_tasks if isinstance(row, dict)]
    if len(lcb_ids) != 200 or len(set(lcb_ids)) != 200 or any(lcb_ids.count(task_id) != 1 for task_id in LCB_HOLD_TASK_IDS):
        raise ValueError("frozen_lcb_manifest_hold_mismatch")

    holds = sorted(HOLD_TASK_IDS)
    tasks = [
        task
        for task in loaded["HumanEval+"] + loaded["MBPP+"]
        if task["task_id"] not in HOLD_TASK_IDS
    ]
    if len(tasks) != 540:
        raise ValueError(f"evalplus_runnable_task_count_mismatch:{len(tasks)}")
    runnable_plus_count = sum(
        len(task["row"]["plus_input"])
        for task in tasks
        if isinstance(task["row"].get("plus_input"), list)
    )
    plus_chunks_at_default = sum(
        math.ceil(len(task["row"]["plus_input"]) / DEFAULT_CHUNK_SIZE)
        for task in tasks
        if isinstance(task["row"].get("plus_input"), list) and task["row"]["plus_input"]
    )
    inventory = {
        "LCB-stdio-200": {
            "manifest_task_count": len(lcb_tasks),
            "manifest_sha256": lcb_manifest_sha256,
            "held_task_ids": sorted(LCB_HOLD_TASK_IDS),
            "reference_backtestable_count": 0,
            "blocked_no_reference_count": len(lcb_tasks) - len(LCB_HOLD_TASK_IDS),
        },
        "EvalPlus": {
            "HumanEval+": len(loaded["HumanEval+"]),
            "MBPP+": len(loaded["MBPP+"]),
            "held_task_ids": holds,
            "runnable_task_count": len(tasks),
            "runnable_plus_input_count": runnable_plus_count,
            "empty_plus_task_ids": [
                task["task_id"] for task in tasks
                if task["plus_coverage"] == "unscored_empty_plus"
            ],
            "default_chunk_size": DEFAULT_CHUNK_SIZE,
            "default_plus_chunk_count": plus_chunks_at_default,
            "default_worker_child_invocation_count": len(tasks) + plus_chunks_at_default,
            "official_baseline_retained": True,
            "derived_revision_comparable_to_official": False,
        },
    }
    return {
        "runnable_tasks": tasks,
        "inventory": inventory,
        "policy": policy,
        "policy_path": policy_path,
        "policy_sha256": file_digest(policy_path),
        "asset_manifest_sha256": manifest_hashes,
        "asset_sha256": asset_hashes,
        "lcb_manifest_path": lcb_path,
    }


def chunk_plus_inputs(plus_input: Any, *, chunk_size: int) -> list[list[dict[str, Any]]]:
    if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
        raise ValueError("chunk_size_must_be_between_1_and_256")
    if plus_input == {} or plus_input == []:
        return []
    if not isinstance(plus_input, list) or any(not isinstance(args, list) for args in plus_input):
        raise ValueError("invalid_evalplus_plus_input_shape")
    cases = [{"index": index, "args": args} for index, args in enumerate(plus_input)]
    return [cases[start : start + chunk_size] for start in range(0, len(cases), chunk_size)]


def _base_assertion_source(row: Mapping[str, Any], benchmark: str) -> str:
    value = row.get("test") if benchmark == "HumanEval+" else row.get("assertion")
    if isinstance(value, list):
        return "\n".join(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"invalid_base_assertions:{row.get('task_id')}")


# Pure input conversion copied from the pinned official EvalPlus source:
# evalplus/data/mbpp.py, git blob 77cca7741add28f8fceb6067f6c98f713c80784f.
# It runs only inside the disposable worker along with the reference code.
_MBPP_DESERIALIZER_SOURCE = r'''
def _mbpp_deserialize_inputs(task_id, inputs):
    task = int(task_id.split("/")[-1])
    if task in [2,116,132,143,222,261,273,394,399,421,424,429,470,560,579,596,616,630,726,740,744,809]:
        return [[tuple(lst) for lst in inp] for inp in inputs]
    if task in [63,64,70,94,120,237,272,299,400,409,417,438,473,614,780]:
        return [[[tuple(lst) for lst in nested] for nested in inp] for inp in inputs]
    if task in [75,413,444,753]:
        return [[[tuple(lst) for lst in inp[0]]] + [inp[1]] for inp in inputs]
    if task in [106,750]:
        return [[inp[0]] + [tuple(inp[1])] for inp in inputs]
    if task == 115:
        return [[[set(item) if isinstance(item,list) and len(item) else {} for item in inp[0]]] for inp in inputs]
    if task == 124:
        return [(float(inp[0]),complex(inp[1])) for inp in inputs]
    if task in [250,405,446,617,720,763,808]:
        return [[tuple(inp[0])] + [inp[1]] for inp in inputs]
    if task in [259,401,445]:
        return [[tuple(row) for row in [[tuple(lst) for lst in nested] for nested in inp]] for inp in inputs]
    if task == 278:
        modified = [[[tuple(item) if isinstance(item,list) else item for item in inp[0]] for _ in [0]] for inp in inputs]
        return [[tuple(lst) for lst in inp] for inp in modified]
    if task == 307:
        return [[tuple(inp[0]),inp[1],inp[2]] for inp in inputs]
    if task == 722:
        return [[{key:tuple(value) for key,value in inp[0].items()}] + inp[1:] for inp in inputs]
    if task == 252:
        return [[complex(inp[0])] for inp in inputs]
    if task in [580,615,791]:
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value,list) else value
        return [tuples(inp) for inp in inputs]
    return inputs
'''


def _worker_source(task: Mapping[str, Any]) -> str:
    row = task["row"]
    source = task["derived_prompt"] + row["canonical_solution"]
    assertions = _base_assertion_source(row, task["benchmark"])
    mbpp = task["benchmark"] == "MBPP+"
    return (
        "import contextlib, io, json, math, pickle, sys\n"
        f"TASK_SOURCE = {json.dumps(source, ensure_ascii=True)}\n"
        f"BASE_ASSERTIONS = {json.dumps(assertions, ensure_ascii=True)}\n"
        f"TASK_ID = {json.dumps(task['task_id'])}\n"
        f"ENTRY_POINT = {json.dumps(row['entry_point'])}\n"
        f"IS_MBPP = {mbpp!r}\n"
        f"MBPP_NOT_NONE_ENTRY_POINTS = {sorted(MBPP_OUTPUT_NOT_NONE_TASKS)!r}\n"
        f"MARKER = {json.dumps(MARKER_PREFIX)}\n"
        + _MBPP_DESERIALIZER_SOURCE
        + r'''
class _Sink:
    def write(self, value): return len(value)
    def flush(self): pass

def _emit(value):
    sys.__stdout__.write(MARKER + json.dumps(value, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()

payload = json.loads(sys.stdin.read())
namespace = {"__name__": "__posttrain_reference__"}
try:
    with contextlib.redirect_stdout(_Sink()), contextlib.redirect_stderr(_Sink()):
        exec(TASK_SOURCE, namespace, namespace)
        function = namespace[ENTRY_POINT]
        if payload["kind"] == "base":
            if IS_MBPP:
                namespace["math"] = math
                exec(BASE_ASSERTIONS, namespace, namespace)
            else:
                exec(BASE_ASSERTIONS, namespace, namespace)
                namespace["check"](function)
            marker = {"kind": "base", "status": "passed"}
        else:
            counts = {"kind":"plus", "planned":len(payload["cases"]), "processed":0,
                      "serialization_ok":0, "serialization_failed":0, "runtime_errors":0,
                      "official_special_normalized":0, "normalized_serialization_ok":0,
                      "normalized_serialization_failed":0,
                      "policy_matches":0, "policy_conflicts":0, "unscored":0}
            for case in payload["cases"]:
                counts["processed"] += 1
                try:
                    args = case["args"]
                    if IS_MBPP:
                        args = _mbpp_deserialize_inputs(TASK_ID, [args])[0]
                    output = function(*args)
                except Exception:
                    counts["runtime_errors"] += 1
                    continue
                try:
                    pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
                    counts["serialization_ok"] += 1
                except Exception:
                    counts["serialization_failed"] += 1
                if IS_MBPP and ENTRY_POINT in MBPP_NOT_NONE_ENTRY_POINTS:
                    # Pinned EvalPlus special oracle: bool outputs are retained;
                    # all other outputs are normalized to `output is not None`.
                    normalized_output = output if isinstance(output, bool) else output is not None
                    counts["official_special_normalized"] += 1
                    try:
                        pickle.dumps(normalized_output, protocol=pickle.HIGHEST_PROTOCOL)
                        counts["normalized_serialization_ok"] += 1
                    except Exception:
                        counts["normalized_serialization_failed"] += 1
                disposition = case["policy"]["disposition"]
                if disposition == "unscored":
                    counts["unscored"] += 1
                elif disposition == "replace_expected":
                    expected = case["policy"]["expected"]
                    try:
                        matches = type(output) is type(expected) and bool(output == expected)
                    except Exception:
                        matches = False
                    if matches:
                        counts["policy_matches"] += 1
                    else:
                        counts["policy_conflicts"] += 1
            marker = counts
except Exception as exc:
    if payload["kind"] == "base":
        marker = {"kind":"base", "status":"failed", "error_type":type(exc).__name__}
    else:
        planned = len(payload.get("cases", []))
        marker = {"kind":"plus", "planned":planned, "processed":0, "serialization_ok":0,
                  "serialization_failed":0, "runtime_errors":0,
                  "official_special_normalized":0, "normalized_serialization_ok":0,
                  "normalized_serialization_failed":0, "policy_matches":0,
                  "policy_conflicts":0, "unscored":0, "error_type":type(exc).__name__}
_emit(marker)
'''
    )


def _policy_decisions(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    row = task["row"]
    raw_plus = row.get("plus_input")
    if task["plus_coverage"] != "scoreable":
        return []
    decisions = []
    for index, args in enumerate(raw_plus):
        has_registered_rule = any(
            rule.get("index") == index for rule in task["task_policy"].get("case_rules", [])
        )
        decision = apply_case_policy(
            task["task_policy"],
            task["task_id"],
            index,
            args,
            digest(args) if has_registered_rule else "",
        )
        if decision["disposition"] == "unscored":
            # Do not attach an oracle value to cases excluded from scoring.
            decision = {"disposition": "unscored"}
        decisions.append(decision)
    return decisions


def prepare_task_run(task: Mapping[str, Any], *, chunk_size: int) -> dict[str, Any]:
    """Build worker source and base/plus JSON stdin batches without executing code."""
    plus_input = task["row"].get("plus_input")
    chunks = chunk_plus_inputs(plus_input, chunk_size=chunk_size) if task["plus_coverage"] == "scoreable" else []
    decisions = _policy_decisions(task)
    stdins = [json.dumps({"kind": "base"}, separators=(",", ":"))]
    groups = [{"kind": "base", "planned_cases": 1, "indices": []}]
    for chunk in chunks:
        cases = []
        for case in chunk:
            cases.append({**case, "policy": decisions[case["index"]]})
        # Python's JSON decoder/encoder pair preserves upstream +/-Infinity as
        # non-standard JSON numeric literals; Mbpp/404 contains one such frozen input.
        stdins.append(json.dumps({"kind": "plus", "cases": cases}, ensure_ascii=False, allow_nan=True, separators=(",", ":")))
        groups.append({"kind": "plus", "planned_cases": len(cases), "indices": [c["index"] for c in cases]})
    if len(stdins) > 4096:
        raise ValueError(f"too_many_task_batch_inputs:{task['task_id']}")
    return {
        "code": _worker_source(task),
        "stdins": stdins,
        "groups": groups,
        "plus_coverage": task["plus_coverage"],
        "plus_input_count": len(plus_input) if isinstance(plus_input, list) else 0,
    }


def _extract_marker(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER_PREFIX):
            try:
                marker = json.loads(line[len(MARKER_PREFIX) :])
            except ValueError:
                return None
            return marker if isinstance(marker, dict) else None
    return None


def _result_state(status: Status) -> str:
    if status == Status.TIMEOUT:
        return "timed_out"
    if status in (Status.INFRA, Status.DATA):
        return "unknown"
    if status == Status.OK:
        return "ok"
    return "execution_failed"


def _case_result_evidence(result: CaseResult) -> dict[str, Any]:
    status = result.status.value if isinstance(result.status, Status) else str(result.status)
    return {
        "status": status,
        "detail": result.detail,
        "physical_executions": result.physical_executions,
        "elapsed_s": result.elapsed_s,
        "exit_code": result.exit_code,
        "truncated": result.truncated,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }


def interpret_worker_result(
    result: CaseResult, *, expected_kind: str, planned_cases: int
) -> dict[str, Any]:
    """Decode a marker conservatively; a timeout or malformed marker is incomplete."""
    unknown = {
        "state": _result_state(result.status),
        "processed": 0,
        "incomplete": planned_cases,
    }
    if result.status != Status.OK:
        return unknown
    marker = _extract_marker(result.stdout)
    if not isinstance(marker, dict) or marker.get("kind") != expected_kind:
        return {**unknown, "state": "incomplete"}
    if expected_kind == "base":
        status = marker.get("status")
        if status not in ("passed", "failed"):
            return {**unknown, "state": "incomplete"}
        return {"state": status, "processed": 1, "incomplete": 0}
    planned = marker.get("planned")
    processed = marker.get("processed")
    counter_names = (
        "serialization_ok",
        "serialization_failed",
        "runtime_errors",
        "official_special_normalized",
        "normalized_serialization_ok",
        "normalized_serialization_failed",
        "policy_matches",
        "policy_conflicts",
        "unscored",
    )
    if planned != planned_cases or type(processed) is not int or not 0 <= processed <= planned_cases:
        return {**unknown, "state": "incomplete"}
    if any(type(marker.get(name)) is not int or marker[name] < 0 for name in counter_names):
        return {**unknown, "state": "incomplete"}
    if (
        marker["serialization_ok"]
        + marker["serialization_failed"]
        + marker["runtime_errors"]
        != processed
        or marker["official_special_normalized"]
        > processed - marker["runtime_errors"]
        or marker["normalized_serialization_ok"]
        + marker["normalized_serialization_failed"]
        != marker["official_special_normalized"]
        or marker["policy_matches"]
        + marker["policy_conflicts"]
        + marker["unscored"]
        > processed - marker["runtime_errors"]
    ):
        return {**unknown, "state": "unknown", "reported_processed": processed}
    return {
        "state": "complete" if processed == planned_cases else "incomplete",
        "processed": processed,
        "incomplete": planned_cases - processed,
        **{name: marker[name] for name in counter_names},
    }


def _safe_task_path(task_id: str) -> str:
    return task_id.replace("/", "-").replace(":", "-")


def _controller_dependency_hashes(root: Path) -> dict[str, str]:
    relative_paths = (
        "scripts/run_eval_reference_backtest.py",
        "src/posttrain/__init__.py",
        "src/posttrain/common.py",
        "src/posttrain/evaluation/__init__.py",
        "src/posttrain/evaluation/evalplus_policy.py",
        "src/posttrain/evaluation/reference_backtest.py",
        "src/posttrain/execution/__init__.py",
        "src/posttrain/execution/contracts.py",
        "src/posttrain/execution/docker_backend.py",
        "src/posttrain/execution/sandboxfusion.py",
    )
    hashes = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"controller_dependency_missing:{relative}")
        hashes[relative] = file_digest(path)
    return hashes


def _run_one_task(
    task: Mapping[str, Any],
    output: Path,
    backend: Any,
    limits: Limits,
    prepared: Mapping[str, Any],
    timeout_s: float,
    attempt: int,
    intent_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    run_id = uuid.uuid4().hex
    record_path = output / f"attempt-{attempt:04d}.json"
    row = task["row"]
    raw_plus = row.get("plus_input")
    record: dict[str, Any] = {
        "benchmark": task["benchmark"],
        "task_id": task["task_id"],
        "attempt": attempt,
        "source_row_sha256": task["row_sha256"],
        "canonical_source_sha256": digest(task["derived_prompt"] + row["canonical_solution"]),
        "base_assertion_sha256": digest(_base_assertion_source(row, task["benchmark"])),
        "started_at": utc_now(),
        "plus_correctness_evidence": "execution_and_serialization_only",
        "plus_coverage": prepared["plus_coverage"],
        "plus_inputs_planned": prepared["plus_input_count"],
        "plus_chunks": [],
    }
    try:
        results = backend.run_batch(
            prepared["code"],
            prepared["stdins"],
            limits,
            run_id,
            timeout_s,
        )
        if not isinstance(results, (list, tuple)) or len(results) != len(prepared["groups"]):
            results = [CaseResult(Status.INFRA, physical_executions=None) for _ in prepared["groups"]]
    except Exception as exc:
        results = [CaseResult(Status.INFRA, physical_executions=None, detail=type(exc).__name__)
                   for _ in prepared["groups"]]

    base_result = interpret_worker_result(results[0], expected_kind="base", planned_cases=1)
    record["base_executor_result"] = _case_result_evidence(results[0])
    record["base_assertion_state"] = base_result["state"]
    record["base_assertion_processed"] = base_result["processed"]
    record["base_assertion_incomplete"] = base_result["incomplete"]
    plus_totals = {
        "planned": prepared["plus_input_count"],
        "processed": 0,
        "incomplete": 0,
        "serialization_ok": 0,
        "serialization_failed": 0,
        "runtime_errors": 0,
        "official_special_normalized": 0,
        "normalized_serialization_ok": 0,
        "normalized_serialization_failed": 0,
        "policy_matches": 0,
        "policy_conflicts": 0,
        "policy_unscored_planned": sum(
            decision["disposition"] == "unscored" for decision in _policy_decisions(task)
        ),
        "unscored": 0,
    }
    for group, result in zip(prepared["groups"][1:], results[1:]):
        interpreted = interpret_worker_result(
            result, expected_kind="plus", planned_cases=group["planned_cases"]
        )
        chunk = {
            "indices": group["indices"],
            "executor_result": _case_result_evidence(result),
            **interpreted,
        }
        record["plus_chunks"].append(chunk)
        for name in (
            "processed", "incomplete", "serialization_ok", "serialization_failed",
            "runtime_errors", "official_special_normalized", "normalized_serialization_ok",
            "normalized_serialization_failed", "policy_matches", "policy_conflicts", "unscored",
        ):
            plus_totals[name] += interpreted.get(name, 0)
    record["plus_execution"] = plus_totals
    record["plus_execution_complete"] = (
        prepared["plus_coverage"] == "scoreable"
        and plus_totals["processed"] == plus_totals["planned"]
        and plus_totals["incomplete"] == 0
    )
    record["policy_comparison_state"] = (
        "conflict" if plus_totals["policy_conflicts"] else
        "matched_registered_cases" if plus_totals["policy_matches"] else
        "unscored_or_not_applicable"
    )
    record["elapsed_s"] = time.monotonic() - started
    record["completed_at"] = utc_now()
    record["controller_timeout_s"] = timeout_s
    write_json(record_path, record)
    return record


def run_reference_backtest(
    repo_root: Path,
    output: Path,
    backend: Any,
    *,
    workers: int = MAX_WORKERS,
    max_wall_hours: float = 7.5,
    task_timeout_seconds: float = 900,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_tasks: int | None = None,
    task_ids: list[str] | None = None,
    execution_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if getattr(backend, "evidence_level", None) != "isolated_execution":
        raise ValueError("isolated_execution_backend_required")
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("workers_must_be_between_1_and_4")
    if isinstance(max_wall_hours, bool) or not isinstance(max_wall_hours, (int, float)) or not math.isfinite(max_wall_hours) or not 0 < max_wall_hours <= MAX_WALL_HOURS:
        raise ValueError("max_wall_hours_must_be_positive_and_at_most_8")
    if isinstance(task_timeout_seconds, bool) or not isinstance(task_timeout_seconds, (int, float)) or not math.isfinite(task_timeout_seconds) or task_timeout_seconds <= 0:
        raise ValueError("task_timeout_seconds_must_be_positive")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
        raise ValueError("chunk_size_must_be_between_1_and_256")

    output = Path(output)
    if output.exists():
        raise FileExistsError("refuse_duplicate_dispatch: output directory already exists")
    plan = build_backtest_plan(repo_root)
    source_tasks = plan["runnable_tasks"]
    source_plan_task_count = len(source_tasks)
    selection_mode = "all"
    if max_tasks is not None and task_ids is not None:
        raise ValueError("max_tasks_and_task_ids_are_mutually_exclusive")
    if task_ids is not None:
        if type(task_ids) is not list or not task_ids or any(type(task_id) is not str or not task_id for task_id in task_ids):
            raise ValueError("task_ids_must_be_a_non_empty_list_of_strings")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate_task_ids")
        source_by_id = {task["task_id"]: task for task in source_tasks}
        held_ids = set(HOLD_TASK_IDS) | set(LCB_HOLD_TASK_IDS)
        held_requested = [task_id for task_id in task_ids if task_id in held_ids]
        if held_requested:
            raise ValueError("held_task_ids:" + ",".join(held_requested))
        unknown_requested = [task_id for task_id in task_ids if task_id not in source_by_id]
        if unknown_requested:
            raise ValueError("unknown_task_ids:" + ",".join(unknown_requested))
        tasks = [source_by_id[task_id] for task_id in task_ids]
        selection_mode = "explicit_task_ids"
    else:
        tasks = source_tasks
    if max_tasks is not None:
        if type(max_tasks) is not int or not 1 <= max_tasks <= len(tasks):
            raise ValueError("max_tasks_out_of_range")
        tasks = tasks[:max_tasks]
        selection_mode = "max_tasks_prefix"
    selected_task_ids = [task["task_id"] for task in tasks]
    output.mkdir(parents=True, exist_ok=False)
    task_index = [
        {
            "benchmark": task["benchmark"],
            "task_id": task["task_id"],
            "source_row_sha256": task["row_sha256"],
            "plus_inputs": len(task["row"].get("plus_input", [])) if isinstance(task["row"].get("plus_input"), list) else 0,
            "plus_coverage": task["plus_coverage"],
        }
        for task in tasks
    ]
    limits = getattr(backend, "limits", Limits())
    identity = {
        "selection_sha256": digest(task_index),
        "source_plan_task_count": source_plan_task_count,
        "selection_mode": selection_mode,
        "selected_task_ids_sha256": digest(selected_task_ids),
        "policy_sha256": plan["policy_sha256"],
        "asset_manifest_sha256": plan["asset_manifest_sha256"],
        "asset_sha256": plan["asset_sha256"],
        "lcb_manifest_sha256": plan["inventory"]["LCB-stdio-200"]["manifest_sha256"],
        "executor_version": str(backend.version),
        "limits": limits.__dict__,
        "workers": workers,
        "task_timeout_seconds": task_timeout_seconds,
        "max_wall_hours": max_wall_hours,
        "chunk_size": chunk_size,
        "pinned_evalplus_commit": PINNED_EVALPLUS_COMMIT,
        "pinned_special_oracle_blob_sha256": PINNED_SPECIAL_ORACLE_BLOB_SHA256,
        "pinned_eval_init_blob_sha256": PINNED_EVAL_INIT_BLOB_SHA256,
        "controller_sha256": file_digest(Path(__file__)),
        "controller_dependency_sha256": _controller_dependency_hashes(Path(repo_root)),
        "external_execution_identity": dict(execution_identity or {}),
    }
    write_json(
        output / "run_manifest.json",
        {
            "run_id": output.name,
            "created_at": utc_now(),
            "identity": identity,
            "inventory": plan["inventory"],
            "selected_tasks": len(tasks),
            "source_plan_tasks": source_plan_task_count,
            "selected_task_ids": selected_task_ids,
            "resume_supported": False,
            "existing_output_policy": "refuse_duplicate_dispatch",
            "evidence_level": "isolated_execution",
            "plus_claim": "execution_and_serialization_only; no independent correctness proof",
            "task_index": task_index,
        },
    )

    task_root = output / "tasks"
    intent_root = output / "intents"
    task_root.mkdir()
    intent_root.mkdir()
    started = time.monotonic()
    deadline = started + float(max_wall_hours) * 3600
    pending = iter(enumerate(tasks))
    futures: dict[concurrent.futures.Future, tuple[int, Mapping[str, Any], Path]] = {}
    task_dirs: dict[int, Path] = {}
    submitted: set[int] = set()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    deadline_hit = False
    try:
        exhausted = False
        while futures or not exhausted:
            while len(futures) < workers and not exhausted:
                remaining = deadline - time.monotonic()
                # Reserve up to 20 seconds for Docker cleanup after a controller timeout.
                if remaining <= 20:
                    deadline_hit = True
                    exhausted = True
                    break
                try:
                    index, task = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                prepared = prepare_task_run(task, chunk_size=chunk_size)
                timeout_s = min(float(task_timeout_seconds), remaining - 20)
                bench_dir = task_root / ("humaneval" if task["benchmark"] == "HumanEval+" else "mbpp")
                bench_dir.mkdir(exist_ok=True)
                task_dir = bench_dir / _safe_task_path(task["task_id"])
                task_dir.mkdir(exist_ok=False)
                task_dirs[index] = task_dir
                intent_path = intent_root / f"{index:04d}-{_safe_task_path(task['task_id'])}.json"
                write_json(
                    intent_path,
                    {
                        "benchmark": task["benchmark"],
                        "task_id": task["task_id"],
                        "attempt": 1,
                        "status": "dispatched",
                        "planned_plus_inputs": prepared["plus_input_count"],
                        "source_row_sha256": task["row_sha256"],
                        "created_at": utc_now(),
                        "task_record_path": str(task_dir / "attempt-0001.json"),
                    },
                )
                future = pool.submit(
                    _run_one_task,
                    task,
                    task_dir,
                    backend,
                    limits,
                    prepared,
                    timeout_s,
                    1,
                    intent_path,
                )
                futures[future] = (index, task, task_dir)
                submitted.add(index)

            if not futures:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                deadline_hit = True
                break
            done, _ = concurrent.futures.wait(
                futures,
                timeout=min(remaining, 1.0),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                futures.pop(future)
                try:
                    future.result()
                except Exception:
                    # The missing attempt is materialized as a controller-error
                    # record below; do not invent a successful task result here.
                    pass

        if futures:
            deadline_hit = True
            done, not_done = concurrent.futures.wait(futures, timeout=20)
            for future in done:
                try:
                    future.result()
                except Exception:
                    pass
            for future in not_done:
                _, task, task_dir = futures[future]
                deadline_record = {
                    "benchmark": task["benchmark"],
                    "task_id": task["task_id"],
                    "attempt": 1,
                    "state": "unknown_at_wall_deadline",
                    "base_assertion_state": "unknown",
                    "plus_correctness_evidence": "execution_and_serialization_only",
                    "source_row_sha256": task["row_sha256"],
                    "plus_execution": {
                        "planned": len(task["row"].get("plus_input", [])) if isinstance(task["row"].get("plus_input"), list) else 0,
                        "processed": 0,
                        "incomplete": len(task["row"].get("plus_input", [])) if isinstance(task["row"].get("plus_input"), list) else 0,
                    },
                    "completed_at": utc_now(),
                }
                write_json(task_dir / "deadline-status.json", deadline_record)
                future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Every selected task gets a durable record, including work not dispatched
    # before the global deadline and any controller-side exception.
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_dir = task_dirs.get(index)
        if task_dir is None:
            bench_dir = task_root / ("humaneval" if task["benchmark"] == "HumanEval+" else "mbpp")
            bench_dir.mkdir(exist_ok=True)
            task_dir = bench_dir / _safe_task_path(task["task_id"])
            task_dir.mkdir(exist_ok=True)
            task_dirs[index] = task_dir
        attempt_path = task_dir / "attempt-0001.json"
        deadline_path = task_dir / "deadline-status.json"
        if attempt_path.is_file():
            results.append(_read_json(attempt_path))
            continue
        if deadline_path.is_file():
            results.append(_read_json(deadline_path))
            continue
        plus_count = len(task["row"].get("plus_input", [])) if isinstance(task["row"].get("plus_input"), list) else 0
        missing_record = {
            "benchmark": task["benchmark"],
            "task_id": task["task_id"],
            "attempt": 1,
            "state": "controller_error" if index in submitted else "not_dispatched_before_deadline",
            "base_assertion_state": "unknown" if index in submitted else "not_started",
            "source_row_sha256": task["row_sha256"],
            "plus_correctness_evidence": "execution_and_serialization_only",
            "plus_coverage": task["plus_coverage"],
            "plus_execution": {"planned": plus_count, "processed": 0, "incomplete": plus_count},
            "completed_at": utc_now(),
        }
        write_json(deadline_path, missing_record)
        results.append(missing_record)

    incomplete_tasks = sum(
        result.get("base_assertion_state") not in ("passed", "failed")
        or result.get("plus_execution", {}).get("incomplete", 0) > 0
        or result.get("state") in ("unknown_at_wall_deadline", "controller_error", "not_dispatched_before_deadline")
        for result in results
    )
    summary = {
        "run_id": output.name,
        "evidence_level": "isolated_execution",
        "selected_tasks": len(tasks),
        "source_plan_tasks": source_plan_task_count,
        "selected_task_ids": selected_task_ids,
        "selected_task_ids_sha256": digest(selected_task_ids),
        "task_records_written": len(results),
        "unfinished_or_unknown_tasks": incomplete_tasks,
        "base_assertions_passed": sum(r.get("base_assertion_state") == "passed" for r in results),
        "base_assertions_failed": sum(r.get("base_assertion_state") == "failed" for r in results),
        "base_assertions_unknown": sum(r.get("base_assertion_state") not in ("passed", "failed") for r in results),
        "plus_inputs_planned": sum(r.get("plus_execution", {}).get("planned", 0) for r in results),
        "plus_inputs_processed": sum(r.get("plus_execution", {}).get("processed", 0) for r in results),
        "plus_inputs_incomplete": sum(r.get("plus_execution", {}).get("incomplete", 0) for r in results),
        "serialization_ok": sum(r.get("plus_execution", {}).get("serialization_ok", 0) for r in results),
        "serialization_failed": sum(r.get("plus_execution", {}).get("serialization_failed", 0) for r in results),
        "runtime_errors": sum(r.get("plus_execution", {}).get("runtime_errors", 0) for r in results),
        "official_special_normalized": sum(r.get("plus_execution", {}).get("official_special_normalized", 0) for r in results),
        "normalized_serialization_ok": sum(r.get("plus_execution", {}).get("normalized_serialization_ok", 0) for r in results),
        "normalized_serialization_failed": sum(r.get("plus_execution", {}).get("normalized_serialization_failed", 0) for r in results),
        "policy_matches": sum(r.get("plus_execution", {}).get("policy_matches", 0) for r in results),
        "policy_conflicts": sum(r.get("plus_execution", {}).get("policy_conflicts", 0) for r in results),
        "policy_unscored_cases_planned": sum(r.get("plus_execution", {}).get("policy_unscored_planned", 0) for r in results),
        "unscored_cases_processed": sum(r.get("plus_execution", {}).get("unscored", 0) for r in results),
        "plus_correctness_evidence": "execution_and_serialization_only",
        "deadline_hit": deadline_hit,
        "wall_time_s": time.monotonic() - started,
        "completed_at": utc_now(),
    }
    write_json(output / "status.json", summary)
    return summary
