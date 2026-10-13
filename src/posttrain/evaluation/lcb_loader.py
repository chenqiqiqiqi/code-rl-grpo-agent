"""Read and verify the frozen LCB stdio set without executing dataset code."""

from __future__ import annotations

import base64
import hashlib
import json
import pickletools
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from posttrain.common import digest
from posttrain.data.schema import TestCase


_PRIVATE_PICKLE_FORMS = {
    ("PROTO", "FRAME", "BINUNICODE", "MEMOIZE", "STOP"),
    ("PROTO", "BINUNICODE", "MEMOIZE", "STOP"),
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
FROZEN_LCB_MANIFEST_SHA256 = "140acae21336805788b6a600a2b0a7a6f1c5d3c36df94b00bd2a3fae16565ba2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _safe_source_name(name: object) -> bool:
    return (isinstance(name, str) and bool(name) and name not in {".", ".."}
            and "/" not in name and "\\" not in name and Path(name).name == name)


def _private_cases(encoded: object) -> list[dict[str, Any]]:
    if not isinstance(encoded, str):
        raise ValueError("invalid_private_test_payload")
    try:
        blob = zlib.decompress(base64.b64decode(encoded, validate=True))
        ops = list(pickletools.genops(blob))
    except (ValueError, zlib.error, struct.error, EOFError) as exc:
        raise ValueError("invalid_private_test_payload") from exc

    names = tuple(op.name for op, _, _ in ops)
    if not ops or names not in _PRIVATE_PICKLE_FORMS or ops[-1][2] != len(blob) - 1:
        raise ValueError(f"unsupported private pickle opcodes: {names}")
    strings = [arg for op, arg, _ in ops if op.name == "BINUNICODE"]
    if len(strings) != 1 or not isinstance(strings[0], str):
        raise ValueError("private_payload_is_not_one_json_string")
    try:
        cases = json.loads(strings[0], object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_private_test_json") from exc
    if not isinstance(cases, list):
        raise ValueError("private_payload_is_not_a_list")
    return cases


def _stdio_cases(cases: object, *, origin: str) -> list[TestCase]:
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"empty_or_invalid_{origin}_tests")
    result: list[TestCase] = []
    for index, case in enumerate(cases):
        if (not isinstance(case, dict)
                or not isinstance(case.get("input"), str)
                or not isinstance(case.get("output"), str)
                or case.get("testtype") != "stdin"):
            raise ValueError(f"invalid_{origin}_case_structure:{index}")
        result.append(TestCase(case["input"], case["output"]))
    return result


def _task_record(task: dict[str, Any], raw: dict[str, Any], line: bytes) -> dict[str, Any]:
    task_id = task["task_id"]
    if task["row_hash"] != _sha(line):
        raise ValueError(f"source_row_hash_mismatch:{task_id}")

    prompt = raw.get("question_content")
    public_text = raw.get("public_test_cases")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"invalid_prompt:{task_id}")
    if not isinstance(public_text, str):
        raise ValueError(f"invalid_public_tests:{task_id}")
    if task["statement_hash"] != digest(prompt):
        raise ValueError(f"statement_hash_mismatch:{task_id}")
    if task["public_test_hash"] != _sha(public_text.encode("utf-8")):
        raise ValueError(f"public_test_hash_mismatch:{task_id}")

    try:
        public_data = json.loads(public_text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid_public_test_json:{task_id}") from exc
    public = _stdio_cases(public_data, origin="public")
    private = _stdio_cases(_private_cases(raw.get("private_test_cases")), origin="private")
    tests = tuple(public + private)
    return {
        "task_id": task_id,
        "prompt": prompt,
        "tests": tests,
        "raw_tests_hash": digest([{"stdin": test.stdin, "stdout": test.stdout}
                                  for test in tests]),
        "raw_test_count": len(tests),
        "public_test_count": len(public),
        "private_test_count": len(private),
        "source_file": task["source_file"],
        "row_hash": task["row_hash"],
        "statement_hash": task["statement_hash"],
        "public_test_hash": task["public_test_hash"],
    }


def load_frozen_lcb(
    root: Path,
    *,
    expected_count: int = 200,
    expected_manifest_sha256: str = FROZEN_LCB_MANIFEST_SHA256,
) -> list[dict[str, Any]]:
    """Load the ordered frozen task set after verifying every bound source hash.

    The private upstream field is a compressed pickle, but only its two audited
    string-only opcode forms are accepted. The payload is decoded with
    ``pickletools`` and JSON; it is never passed to ``pickle.loads``.
    """
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("invalid_expected_task_count")
    if (not isinstance(expected_manifest_sha256, str)
            or not _SHA256_PATTERN.fullmatch(expected_manifest_sha256)):
        raise ValueError("invalid_expected_manifest_sha256")
    root = Path(root)
    manifest_path = root / "manifests" / "lcb_stdio_200.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("cannot_read_frozen_lcb_manifest") from exc
    if _sha(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("frozen_manifest_file_sha256_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("cannot_read_frozen_lcb_manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("invalid_frozen_lcb_manifest")

    stated_hash = manifest.get("manifest_hash")
    try:
        calculated_hash = digest({key: value for key, value in manifest.items()
                                  if key != "manifest_hash"})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_frozen_lcb_manifest") from exc
    if not isinstance(stated_hash, str) or stated_hash != calculated_hash:
        raise ValueError("manifest_hash_mismatch")

    tasks = manifest.get("tasks")
    if (type(manifest.get("task_count")) is not int
            or manifest["task_count"] != expected_count
            or not isinstance(tasks, list) or len(tasks) != expected_count):
        raise ValueError("manifest_task_count_mismatch")
    task_ids: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("invalid_manifest_task")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("invalid_manifest_task_id")
        if task_id in selected:
            raise ValueError(f"duplicate_manifest_task_id:{task_id}")
        for field in ("source_file", "row_hash", "statement_hash", "public_test_hash"):
            value = task.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid_manifest_{field}:{task_id}")
        for field in ("row_hash", "statement_hash", "public_test_hash"):
            if not _SHA256_PATTERN.fullmatch(task[field]):
                raise ValueError(f"invalid_manifest_{field}:{task_id}")
        task_ids.append(task_id)
        selected[task_id] = task

    source_entries = manifest.get("source_files")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("invalid_manifest_source_files")
    sources: dict[str, dict[str, Any]] = {}
    for source in source_entries:
        if not isinstance(source, dict):
            raise ValueError("invalid_manifest_source_file")
        name = source.get("file")
        size = source.get("size")
        sha256 = source.get("sha256")
        if not _safe_source_name(name):
            raise ValueError("invalid_manifest_source_path")
        if name in sources:
            raise ValueError(f"duplicate_manifest_source_file:{name}")
        if type(size) is not int or size < 0 or not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
            raise ValueError(f"invalid_manifest_source_identity:{name}")
        sources[name] = source
    for task_id, task in selected.items():
        if task["source_file"] not in sources:
            raise ValueError(f"unlisted_task_source_file:{task_id}")

    found: dict[str, dict[str, Any]] = {}
    source_root = root / "data" / "raw" / "lcb"
    for name, source in sources.items():
        path = source_root / name
        try:
            size_on_disk = path.stat().st_size
            stream = path.open("rb")
        except OSError as exc:
            raise ValueError(f"source_file_unavailable:{name}") from exc
        if size_on_disk != source["size"]:
            stream.close()
            raise ValueError(f"source_size_mismatch:{name}")

        source_hash = hashlib.sha256()
        bytes_read = 0
        with stream:
            for line in stream:
                source_hash.update(line)
                bytes_read += len(line)
                try:
                    raw = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
                except (UnicodeError, ValueError) as exc:
                    raise ValueError(f"invalid_source_jsonl_row:{name}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"invalid_source_jsonl_row:{name}")
                platform = raw.get("platform")
                question_id = raw.get("question_id")
                if not isinstance(platform, str) or not isinstance(question_id, str):
                    continue
                task_id = f"{platform}:{question_id}"
                task = selected.get(task_id)
                if task is None:
                    continue
                if task["source_file"] != name:
                    raise ValueError(f"task_source_file_mismatch:{task_id}")
                if task_id in found:
                    raise ValueError(f"duplicate_source_task_row:{task_id}")
                found[task_id] = _task_record(task, raw, line)

        if bytes_read != source["size"] or source_hash.hexdigest() != source["sha256"]:
            raise ValueError(f"source_identity_mismatch:{name}")

    missing = [task_id for task_id in task_ids if task_id not in found]
    if missing:
        raise ValueError(f"frozen_task_coverage_mismatch:missing={len(missing)}")
    if len(found) != expected_count:
        raise ValueError("frozen_task_coverage_mismatch")
    return [found[task_id] for task_id in task_ids]
