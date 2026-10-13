from __future__ import annotations

import base64
import hashlib
import importlib
import json
import pickle
import struct
import zlib

import pytest

from posttrain.common import digest
from posttrain.data.schema import TestCase as Fixture


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private_blob(cases: list[dict]) -> str:
    """Encode the audit's one-JSON-string pickle form without pickle.dumps."""
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pickle_bytes = b"\x80\x04X" + struct.pack("<I", len(payload)) + payload + b"\x94."
    return base64.b64encode(zlib.compress(pickle_bytes)).decode("ascii")


def _case(stdin: str, stdout: str) -> dict:
    return {"input": stdin, "output": stdout, "testtype": "stdin"}


def _make_fixture(
    root,
    *,
    task_specs: list[tuple[str, str, list[dict], list[dict]]] | None = None,
    source_rows: list[dict] | None = None,
    private_override: str | None = None,
) -> dict:
    if task_specs is None:
        task_specs = [
            ("atcoder:synthetic_b", "Prompt B", [_case("same\n", "B\n")],
             [_case("same\n", "B\n")]),
            ("atcoder:synthetic_a", "Prompt A", [_case("A\n", "a\n")],
             [_case("B\n", "b\n")]),
        ]

    raw_by_id: dict[str, dict] = {}
    task_by_id: dict[str, dict] = {}
    for task_id, prompt, public_cases, private_cases in task_specs:
        platform, question_id = task_id.split(":", 1)
        public_text = json.dumps(public_cases, ensure_ascii=False, separators=(",", ":"))
        raw = {
            "platform": platform,
            "question_id": question_id,
            "question_content": prompt,
            "public_test_cases": public_text,
            "private_test_cases": private_override or _private_blob(private_cases),
        }
        line = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        raw_by_id[task_id] = {"row": raw, "line": line}
        task_by_id[task_id] = {
            "task_id": task_id,
            "source_file": "synthetic.jsonl",
            "row_hash": _sha(line),
            "statement_hash": digest(prompt),
            "public_test_hash": _sha(public_text.encode("utf-8")),
        }

    rows = source_rows if source_rows is not None else [
        raw_by_id[task_id]["row"] for task_id, *_ in reversed(task_specs)
    ]
    lines = []
    for raw in rows:
        task_id = f"{raw['platform']}:{raw['question_id']}"
        lines.append(raw_by_id[task_id]["line"])
    source_bytes = b"".join(lines)

    source_dir = root / "data" / "raw" / "lcb"
    source_dir.mkdir(parents=True)
    (source_dir / "synthetic.jsonl").write_bytes(source_bytes)
    (root / "manifests").mkdir()

    manifest = {
        "task_count": len(task_specs),
        "tasks": [task_by_id[task_id] for task_id, *_ in task_specs],
        "source_files": [{
            "file": "synthetic.jsonl",
            "size": len(source_bytes),
            "sha256": _sha(source_bytes),
        }],
    }
    _write_manifest(root, manifest)
    return manifest


def _write_manifest(root, manifest: dict) -> None:
    manifest["manifest_hash"] = digest({key: value for key, value in manifest.items()
                                       if key != "manifest_hash"})
    (root / "manifests" / "lcb_stdio_200.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def _manifest_sha(root) -> str:
    return _sha((root / "manifests" / "lcb_stdio_200.json").read_bytes())


def _loader():
    try:
        module = importlib.import_module("posttrain.evaluation.lcb_loader")
        return getattr(module, "load_frozen_lcb", None)
    except ImportError:
        return None


def _load(root, expected_manifest_sha256: str, expected_count: int = 200):
    loader = _loader()
    assert callable(loader), "posttrain.evaluation.lcb_loader.load_frozen_lcb is missing"
    return loader(root, expected_count=expected_count,
                  expected_manifest_sha256=expected_manifest_sha256)


def test_load_keeps_manifest_order_and_repeated_raw_tests(tmp_path):
    _make_fixture(tmp_path)

    records = _load(tmp_path, _manifest_sha(tmp_path), expected_count=2)

    assert [record["task_id"] for record in records] == [
        "atcoder:synthetic_b", "atcoder:synthetic_a"
    ]
    first = records[0]
    assert first["prompt"] == "Prompt B"
    assert first["tests"] == (Fixture("same\n", "B\n"), Fixture("same\n", "B\n"))
    assert first["raw_tests_hash"] == digest([
        {"stdin": "same\n", "stdout": "B\n"},
        {"stdin": "same\n", "stdout": "B\n"},
    ])


def test_default_identity_rejects_a_different_manifest(tmp_path):
    _make_fixture(tmp_path)
    loader = _loader()

    with pytest.raises(ValueError, match="frozen_manifest_file_sha256_mismatch"):
        loader(tmp_path, expected_count=2)


def test_rejects_pickle_payload_with_unapproved_opcodes_without_unpickling(tmp_path, monkeypatch):
    malicious = b"\x80\x02cbuiltins\neval\nX\x03\x00\x00\x001+1\x85R."
    encoded = base64.b64encode(zlib.compress(malicious)).decode("ascii")
    _make_fixture(tmp_path, private_override=encoded)
    monkeypatch.setattr(pickle, "loads", lambda *_args, **_kwargs: pytest.fail("pickle.loads called"))

    with pytest.raises(ValueError, match="unsupported private pickle opcodes"):
        _load(tmp_path, _manifest_sha(tmp_path), expected_count=2)


def test_rejects_missing_and_repeated_selected_source_rows(tmp_path):
    manifest = _make_fixture(tmp_path)
    source_path = tmp_path / "data" / "raw" / "lcb" / "synthetic.jsonl"
    source_path.write_bytes(b"")
    manifest["source_files"][0]["size"] = 0
    manifest["source_files"][0]["sha256"] = _sha(b"")
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="coverage|missing"):
        _load(tmp_path, _manifest_sha(tmp_path), expected_count=2)

    manifest = _make_fixture(tmp_path / "repeated")
    source_path = tmp_path / "repeated" / "data" / "raw" / "lcb" / "synthetic.jsonl"
    source_path.write_bytes(source_path.read_bytes() + source_path.read_bytes().splitlines(keepends=True)[0])
    manifest["source_files"][0]["size"] = source_path.stat().st_size
    manifest["source_files"][0]["sha256"] = _sha(source_path.read_bytes())
    _write_manifest(tmp_path / "repeated", manifest)
    with pytest.raises(ValueError, match="duplicate|coverage|multiple"):
        _load(tmp_path / "repeated", _manifest_sha(tmp_path / "repeated"), expected_count=2)


@pytest.mark.parametrize("drift", [
    "manifest_bytes_sha256", "manifest_hash", "source_sha256", "source_size", "row_hash",
    "statement_hash", "public_test_hash",
])
def test_rejects_frozen_identity_hash_drift(tmp_path, drift):
    manifest = _make_fixture(tmp_path)
    pinned_fixture_hash = _manifest_sha(tmp_path)
    if drift == "manifest_bytes_sha256":
        manifest["tasks"][0]["task_id"] = "atcoder:tampered"
        (tmp_path / "manifests" / "lcb_stdio_200.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        expected_manifest_sha256 = pinned_fixture_hash
    elif drift == "manifest_hash":
        manifest["tasks"][0]["task_id"] = "atcoder:tampered"
        (tmp_path / "manifests" / "lcb_stdio_200.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        expected_manifest_sha256 = _manifest_sha(tmp_path)
    else:
        if drift == "source_sha256":
            manifest["source_files"][0]["sha256"] = "0" * 64
        elif drift == "source_size":
            manifest["source_files"][0]["size"] += 1
        else:
            manifest["tasks"][0][drift] = "0" * 64
        _write_manifest(tmp_path, manifest)
        expected_manifest_sha256 = _manifest_sha(tmp_path)

    with pytest.raises(ValueError):
        _load(tmp_path, expected_manifest_sha256, expected_count=2)


def test_rejects_duplicate_manifest_task_ids(tmp_path):
    manifest = _make_fixture(tmp_path)
    manifest["tasks"][1] = dict(manifest["tasks"][0])
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unique|duplicate"):
        _load(tmp_path, _manifest_sha(tmp_path), expected_count=2)
