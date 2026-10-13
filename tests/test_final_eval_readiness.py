from __future__ import annotations

import hashlib
import json

from posttrain.evaluation.readiness import build_readiness_report


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _observation(content: bytes | None) -> dict:
    if content is None:
        return {"exists": False, "sha256": None, "size": None}
    return {
        "exists": True,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _fixture() -> tuple[dict[str, bytes | None], dict[str, dict]]:
    paths = {
        "lcb_manifest": "manifests/lcb_stdio_200.json",
        "he_manifest": "manifests/humaneval_plus.json",
        "mbpp_manifest": "manifests/mbpp_plus.json",
        "taco_manifest": "manifests/taco_train_sources.json",
        "lcb_raw": "data/raw/lcb/test5.jsonl",
        "he_raw": "data/raw/HumanEvalPlus-v0.1.10.jsonl.gz",
        "mbpp_raw": "data/raw/MbppPlus-v0.2.0.jsonl.gz",
        "current_candidate": "data/processed/stage1-completion-cleaning-local-v3/reliable-candidates.jsonl",
    }
    content = {
        "lcb_raw": b"synthetic-lcb-source",
        "he_raw": b"synthetic-he-plus-source",
        "mbpp_raw": b"synthetic-mbpp-plus-source",
        "candidate": b"synthetic-current-candidates",
        "taco_manifest": b"synthetic-taco-source-manifest",
    }
    manifests = {
        "lcb": {
            "benchmark": "LCB-stdio-200",
            "task_count": 1,
            "source_files": [{"file": "test5.jsonl", "size": len(content["lcb_raw"]),
                              "sha256": hashlib.sha256(content["lcb_raw"]).hexdigest()}],
        },
        "humaneval_plus": {
            "benchmark": "HumanEval+",
            "task_count": 1,
            "file": "HumanEvalPlus-v0.1.10.jsonl.gz",
            "file_sha256": hashlib.sha256(content["he_raw"]).hexdigest(),
        },
        "mbpp_plus": {
            "benchmark": "MBPP+",
            "task_count": 1,
            "file": "MbppPlus-v0.2.0.jsonl.gz",
            "file_sha256": hashlib.sha256(content["mbpp_raw"]).hexdigest(),
        },
    }
    documents: dict[str, bytes | None] = {
        "manifests/lcb_stdio_200.json": _json_bytes(manifests["lcb"]),
        "manifests/humaneval_plus.json": _json_bytes(manifests["humaneval_plus"]),
        "manifests/mbpp_plus.json": _json_bytes(manifests["mbpp_plus"]),
        "reports/evalset-quality-audit/issue_evidence.json": _json_bytes(
            {"lcb_arc188_c": {"case_hash": "case"}, "mbpp": {"Mbpp/793": {}}}
        ),
        "reports/evalset-quality-audit/train_exclusion_candidates.json": _json_bytes({
            "final_manifest_sha256": {},
            "taco_source_manifest_sha256": hashlib.sha256(content["taco_manifest"]).hexdigest(),
            "snapshots": [{
                "path": paths["current_candidate"],
                "sha256": hashlib.sha256(content["candidate"]).hexdigest(),
                "records": 1,
            }, {
                "path": "data/processed/taco-training-pool-v1/problems.jsonl",
                "sha256": "0" * 64,
                "records": 1,
            }],
            "source_key_hits": 0,
            "exact_statement_hits": 0,
            "normalized_statement_hits": 0,
            "near_statement_hits": 0,
            "training_side_ids_to_exclude": 0,
        }),
    }
    integrity = {
        "scope": "static_structure_only_no_dataset_code_or_model_execution",
        "lcb": {
            "manifest_sha256": hashlib.sha256(documents[paths["lcb_manifest"]]).hexdigest(),
            "selected_found": 1,
            "counts": {"source_rows": 1, "public_tests": 2, "private_tests": 3},
            "source_files": {"test5.jsonl": _observation(content["lcb_raw"])},
            "issues": [],
        },
    }
    for name, raw_key, manifest_key in [
        ("humaneval_plus", "he_raw", "he_manifest"),
        ("mbpp_plus", "mbpp_raw", "mbpp_manifest"),
    ]:
        manifest_path = paths[manifest_key]
        integrity[name] = {
            "manifest_sha256": hashlib.sha256(documents[manifest_path]).hexdigest(),
            "source_file": manifests[name]["file"],
            "sha256": hashlib.sha256(content[raw_key]).hexdigest(),
            "size": len(content[raw_key]),
            "task_count": 1,
            "expected_task_count": 1,
            "base_input_count": 2,
            "plus_input_count": 4,
            "issues": [],
            "empty_plus_input_task_ids": [],
            "prompt_without_def_signature_count": 0,
        }
    integrity["humaneval_plus"]["missing"] = []
    integrity["humaneval_plus"]["nonunique"] = {}
    integrity["mbpp_plus"]["missing"] = []
    integrity["mbpp_plus"]["nonunique"] = {}
    integrity["mbpp_plus"]["invalid_plus_input_count"] = 0
    integrity_bytes = _json_bytes(integrity)
    evidence_bytes = documents["reports/evalset-quality-audit/issue_evidence.json"]
    documents["reports/evalset-quality-audit/integrity.json"] = integrity_bytes
    documents["reports/evalset-quality-audit/PROGRESS.md"] = (
        "`integrity.json` SHA256 `" + hashlib.sha256(integrity_bytes).hexdigest() + "`; "
        "`issue_evidence.json` SHA256 `" + hashlib.sha256(evidence_bytes).hexdigest() + "`"
    ).encode()
    exclusion = json.loads(documents["reports/evalset-quality-audit/train_exclusion_candidates.json"])
    exclusion["final_manifest_sha256"] = {
        name: hashlib.sha256(documents[path]).hexdigest()
        for name, path in [
            ("lcb_stdio_200.json", paths["lcb_manifest"]),
            ("humaneval_plus.json", paths["he_manifest"]),
            ("mbpp_plus.json", paths["mbpp_manifest"]),
        ]
    }
    documents["reports/evalset-quality-audit/train_exclusion_candidates.json"] = _json_bytes(exclusion)
    observations = {
        paths["lcb_manifest"]: _observation(documents[paths["lcb_manifest"]]),
        paths["he_manifest"]: _observation(documents[paths["he_manifest"]]),
        paths["mbpp_manifest"]: _observation(documents[paths["mbpp_manifest"]]),
        paths["taco_manifest"]: _observation(content["taco_manifest"]),
        paths["lcb_raw"]: _observation(content["lcb_raw"]),
        paths["he_raw"]: _observation(content["he_raw"]),
        paths["mbpp_raw"]: _observation(content["mbpp_raw"]),
        paths["current_candidate"]: _observation(content["candidate"]),
    }
    return documents, observations


def _evaluate(documents=None, observations=None) -> dict:
    default_documents, default_observations = _fixture()
    return build_readiness_report(
        documents if documents is not None else default_documents,
        observations if observations is not None else default_observations,
    )


def test_source_byte_drift_invalidates_cached_audit() -> None:
    documents, observations = _fixture()
    observations["data/raw/lcb/test5.jsonl"] = _observation(b"changed source bytes")

    report = _evaluate(documents, observations)

    assert report["cache"]["reusable"] is False
    assert report["benchmarks"]["lcb"]["cache_status"] == "source_drift"
    assert any(x["code"] == "source_bytes_drifted" for x in report["benchmarks"]["lcb"]["blockers"])


def test_missing_audit_report_is_reported_without_claiming_readiness() -> None:
    documents, observations = _fixture()
    documents["reports/evalset-quality-audit/integrity.json"] = None

    report = _evaluate(documents, observations)

    assert report["cache"]["reusable"] is False
    assert report["audit_reports"]["integrity"]["status"] == "missing"
    assert all(value["status"] != "ready" for value in report["benchmarks"].values())


def test_task_count_drift_against_frozen_manifest_is_explicit() -> None:
    documents, observations = _fixture()
    manifest = json.loads(documents["manifests/humaneval_plus.json"])
    manifest["task_count"] = 2
    documents["manifests/humaneval_plus.json"] = _json_bytes(manifest)
    observations["manifests/humaneval_plus.json"] = _observation(
        documents["manifests/humaneval_plus.json"]
    )

    report = _evaluate(documents, observations)

    benchmark = report["benchmarks"]["humaneval_plus"]
    assert benchmark["coverage"]["count_consistent"] is False
    assert any(x["code"] == "task_count_drift" for x in benchmark["blockers"])


def test_tampered_summary_fails_report_digest_check() -> None:
    documents, observations = _fixture()
    integrity = json.loads(documents["reports/evalset-quality-audit/integrity.json"])
    integrity["lcb"]["selected_found"] = 99
    documents["reports/evalset-quality-audit/integrity.json"] = _json_bytes(integrity)

    report = _evaluate(documents, observations)

    assert report["audit_reports"]["integrity"]["status"] == "digest_mismatch"
    assert report["cache"]["reusable"] is False


def test_missing_historical_snapshot_does_not_invalidate_baseline_audit_reuse() -> None:
    report = _evaluate()

    candidate = report["cache"]["baseline_candidate_snapshot"]
    assert candidate["status"] == "matched"
    assert report["cache"]["reusable"] is True
    assert report["cache"]["missing_historical_snapshots"] == [
        "data/processed/taco-training-pool-v1/problems.jsonl"
    ]
    assert report["full_review"] is False
    assert all(value["status"] == "blocked" for value in report["benchmarks"].values())
