from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

AUDIT = "reports/evalset-quality-audit"
INTEGRITY = f"{AUDIT}/integrity.json"
EVIDENCE = f"{AUDIT}/issue_evidence.json"
EXCLUSIONS = f"{AUDIT}/train_exclusion_candidates.json"
PROGRESS = f"{AUDIT}/PROGRESS.md"
MANIFESTS = {
    "lcb": "manifests/lcb_stdio_200.json",
    "humaneval_plus": "manifests/humaneval_plus.json",
    "mbpp_plus": "manifests/mbpp_plus.json",
}
BASELINE_CANDIDATE = "data/processed/stage1-completion-cleaning-local-v3/reliable-candidates.jsonl"
TACO_MANIFEST = "manifests/taco_train_sources.json"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(documents: Mapping[str, bytes | None], path: str) -> dict | None:
    try:
        value = json.loads(documents[path]) if documents.get(path) is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _pin_status(documents: Mapping[str, bytes | None], progress: str | None, path: str) -> tuple[dict, dict | None]:
    value = _json(documents, path)
    content = documents.get(path)
    if content is None:
        return {"status": "missing"}, None
    if value is None:
        return {"status": "invalid_json"}, None
    name = path.rsplit("/", 1)[-1]
    match = re.search(rf"`{re.escape(name)}`\s+SHA256\s+`([0-9a-f]{{64}})`", progress or "", re.I)
    if not match:
        return {"status": "digest_pin_missing"}, value
    actual = sha256_bytes(content)
    expected = match.group(1).lower()
    return {"status": "verified" if actual == expected else "digest_mismatch",
            "sha256": actual, "expected_sha256": expected}, value


def build_readiness_report(documents: Mapping[str, bytes | None], observations: Mapping[str, dict]) -> dict:
    """Assess cached reports and byte identities; never parses benchmark payloads."""
    progress_bytes = documents.get(PROGRESS)
    progress = progress_bytes.decode("utf-8", errors="replace") if progress_bytes else None
    integrity_status, integrity = _pin_status(documents, progress, INTEGRITY)
    evidence_status, evidence = _pin_status(documents, progress, EVIDENCE)
    exclusion = _json(documents, EXCLUSIONS)
    reports = {
        "progress": {"status": "available" if progress else "missing"},
        "integrity": integrity_status,
        "issue_evidence": evidence_status,
        "train_exclusion_candidates": {"status": "available" if exclusion else "missing_or_invalid"},
    }
    issues: list[dict] = []
    sources: dict[str, list[dict]] = {}
    benchmarks: dict[str, dict] = {}

    def add_issue(code: str, path: str, detail: str) -> None:
        issues.append({"code": code, "path": path, "detail": detail})

    for name, status in (("integrity", integrity_status), ("issue_evidence", evidence_status)):
        if status["status"] != "verified":
            add_issue("audit_report_not_verified", f"{AUDIT}/{name}.json", status["status"])
    if not progress:
        add_issue("audit_progress_missing", PROGRESS, "Digest pins are unavailable.")
    if not exclusion:
        add_issue("dedup_report_missing", EXCLUSIONS, "Prior overlap evidence is unavailable.")

    for benchmark, manifest_path in MANIFESTS.items():
        audit = (integrity or {}).get(benchmark, {})
        manifest = _json(documents, manifest_path)
        manifest_bytes = documents.get(manifest_path)
        manifest_obs = observations.get(manifest_path, {})
        manifest_sha = sha256_bytes(manifest_bytes) if manifest_bytes is not None else None
        checks: list[dict] = []
        if manifest is None or not manifest_obs.get("exists"):
            checks.append({"path": manifest_path, "status": "missing"})
        elif manifest_obs.get("sha256") != manifest_sha:
            checks.append({"path": manifest_path, "status": "observation_mismatch"})
        else:
            expected = audit.get("manifest_sha256")
            checks.append({"path": manifest_path,
                           "status": "matched" if manifest_sha == expected else "drifted",
                           "expected_sha256": expected, "actual_sha256": manifest_sha})

        if benchmark == "lcb":
            raw_files = audit.get("source_files", {})
            for filename, expected in raw_files.items() if isinstance(raw_files, dict) else []:
                if not isinstance(filename, str) or "/" in filename or "\\" in filename:
                    checks.append({"path": str(filename), "status": "invalid_source_name"})
                    continue
                raw_path = f"data/raw/lcb/{filename}"
                actual = observations.get(raw_path, {})
                expected = expected if isinstance(expected, dict) else {}
                match = actual.get("exists") and actual.get("sha256") == expected.get("sha256") and actual.get("size") == expected.get("size")
                checks.append({"path": raw_path, "status": "matched" if match else "drifted",
                               "expected_sha256": expected.get("sha256"), "actual_sha256": actual.get("sha256")})
        else:
            filename = audit.get("source_file")
            raw_path = f"data/raw/{filename}" if isinstance(filename, str) and "/" not in filename and "\\" not in filename else "data/raw"
            actual = observations.get(raw_path, {})
            raw_match = (actual.get("exists") and actual.get("sha256") == audit.get("sha256")
                         and actual.get("size") == audit.get("size"))
            manifest_match = manifest is not None and manifest.get("file_sha256") == audit.get("sha256")
            checks.append({"path": raw_path, "status": "matched" if raw_match and manifest_match else "drifted",
                           "expected_sha256": audit.get("sha256"), "actual_sha256": actual.get("sha256")})

        manifest_count = manifest.get("task_count") if manifest else None
        audited_count = audit.get("selected_found" if benchmark == "lcb" else "task_count")
        expected_count = manifest_count if benchmark == "lcb" else audit.get("expected_task_count")
        task_rows = manifest.get("tasks") if manifest else None
        count_ok = (isinstance(manifest_count, int) and manifest_count == audited_count
                    and (expected_count is None or expected_count == manifest_count)
                    and (not isinstance(task_rows, list) or len(task_rows) == manifest_count))
        if not count_ok:
            add_issue("task_count_drift", manifest_path, f"{benchmark} report and manifest counts disagree.")
        for row in checks:
            if row["status"] != "matched":
                add_issue("source_bytes_not_verified", row["path"], row["status"])

        if benchmark == "lcb":
            counts = audit.get("counts", {})
            coverage = {"task_count": audited_count, "expected_task_count": manifest_count,
                        "source_rows": counts.get("source_rows"), "public_tests": counts.get("public_tests"),
                        "private_tests": counts.get("private_tests"),
                        "structural_issue_count": len(audit.get("issues", [])), "count_consistent": count_ok}
        else:
            coverage = {"task_count": audit.get("task_count"), "expected_task_count": audit.get("expected_task_count"),
                        "base_input_count": audit.get("base_input_count"), "plus_input_count": audit.get("plus_input_count"),
                        "structural_issue_count": len(audit.get("issues", [])),
                        "empty_plus_task_count": len(audit.get("empty_plus_input_task_ids", [])),
                        "prompt_without_signature_count": audit.get("prompt_without_def_signature_count"),
                        "count_consistent": count_ok}
        cache_status = "verified" if count_ok and all(x["status"] == "matched" for x in checks) else "source_drift"
        blockers = []
        if not count_ok:
            blockers.append({"code": "task_count_drift", "detail": "Manifest task counts disagree with the cached summary."})
        if any(x["status"] in {"drifted", "missing", "observation_mismatch"} for x in checks):
            blockers.append({"code": "source_bytes_drifted", "detail": "Current manifest or raw source bytes differ from the cached audit."})
        benchmarks[benchmark] = {"status": "blocked", "cache_status": cache_status,
                                 "coverage": coverage, "source_checks": checks,
                                 "full_review": False, "blockers": blockers}
        sources[benchmark] = checks

    candidate = {"path": BASELINE_CANDIDATE, "status": "unavailable"}
    missing_history: list[str] = []
    if exclusion:
        rows = exclusion.get("snapshots", [])
        rows = [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []
        row = next((x for x in rows if x.get("path") == BASELINE_CANDIDATE), None)
        actual = observations.get(BASELINE_CANDIDATE, {})
        if row:
            matched = actual.get("exists") and actual.get("sha256") == row.get("sha256")
            candidate = {"path": BASELINE_CANDIDATE, "status": "matched" if matched else "drifted",
                         "expected_sha256": row.get("sha256"), "actual_sha256": actual.get("sha256"),
                         "records_in_prior_report": row.get("records")}
            if not matched:
                add_issue("baseline_candidate_snapshot_not_reusable", BASELINE_CANDIDATE, "Bytes differ from the old overlap report.")
        else:
            add_issue("baseline_candidate_snapshot_unlisted", BASELINE_CANDIDATE, "No matching snapshot in old overlap report.")
        for item in rows:
            path = item.get("path")
            if path and path != BASELINE_CANDIDATE and not observations.get(path, {}).get("exists"):
                missing_history.append(path)
        taco_expected = exclusion.get("taco_source_manifest_sha256")
        taco_actual = observations.get(TACO_MANIFEST, {})
        taco_check = {"path": TACO_MANIFEST,
                      "status": "matched" if taco_actual.get("exists") and taco_actual.get("sha256") == taco_expected else "drifted",
                      "expected_sha256": taco_expected, "actual_sha256": taco_actual.get("sha256")}
        sources["training_pool"] = [taco_check]
        if taco_check["status"] != "matched":
            add_issue("taco_source_manifest_not_verified", TACO_MANIFEST, taco_check["status"])
        expected_manifests = exclusion.get("final_manifest_sha256", {})
        for key, path in (("lcb_stdio_200.json", MANIFESTS["lcb"]),
                          ("humaneval_plus.json", MANIFESTS["humaneval_plus"]),
                          ("mbpp_plus.json", MANIFESTS["mbpp_plus"])):
            if observations.get(path, {}).get("sha256") != expected_manifests.get(key):
                add_issue("dedup_eval_manifest_drift", path, "Manifest differs from the old overlap report.")

    blockers = {
        "lcb": [
            {"code": "isolated_stdio_execution_unaccepted", "status": "current", "detail": "The ARC188_C checker has synthetic acceptance, but no real isolated benchmark execution is accepted.", "evidence": "reports/evalset-remediation/lcb/PROGRESS.md"},
            {"code": "arc184_c_prompt_supplement_blocked", "status": "current", "detail": "The missing diagram has no verified, hash-bound text supplement.", "evidence": "reports/evalset-remediation/lcb/PROGRESS.md"},
        ],
        "humaneval_plus": [
            {"code": "evalplus_remote_byte_identity_unverified", "status": "current", "detail": "Official remote release bytes could not be independently verified.", "evidence": "reports/evalset-remediation/upstream-byte-verification.json"},
            {"code": "isolated_function_execution_unaccepted", "status": "current", "detail": "No real isolated function-task execution acceptance is recorded.", "evidence": PROGRESS},
            {"code": "known_anomaly_policy_pending", "status": "baseline_only", "detail": "The prior audit listed prompt anomalies; this cached baseline does not replace current policy readback.", "evidence": f"{AUDIT}/PROGRESS.md"},
        ],
        "mbpp_plus": [
            {"code": "evalplus_remote_byte_identity_unverified", "status": "current", "detail": "Official remote release bytes could not be independently verified.", "evidence": "reports/evalset-remediation/upstream-byte-verification.json"},
            {"code": "isolated_function_execution_unaccepted", "status": "current", "detail": "No real isolated function-task execution acceptance is recorded.", "evidence": PROGRESS},
            {"code": "known_anomaly_policy_pending", "status": "baseline_only", "detail": "The prior audit listed input/oracle/test anomalies; this cached baseline does not replace current policy readback.", "evidence": f"{AUDIT}/PROGRESS.md"},
        ],
    }
    for key in benchmarks:
        benchmarks[key]["blockers"].extend(blockers[key])
        if integrity_status["status"] != "verified" or evidence_status["status"] != "verified":
            benchmarks[key]["blockers"].append({"code": "audit_cache_unverified", "status": "current", "detail": "Prior audit report digest check failed.", "evidence": AUDIT})

    all_sources_match = all(x["status"] == "matched" for group in sources.values() for x in group)
    pins_match = integrity_status["status"] == evidence_status["status"] == "verified"
    counts_match = all(x["coverage"]["count_consistent"] for x in benchmarks.values())
    reusable = bool(pins_match and exclusion and all_sources_match and counts_match and candidate["status"] == "matched")
    evidence_coverage = {"lcb_finding_groups": int(bool(evidence and "lcb_arc188_c" in evidence)),
                         "mbpp_task_findings": len(evidence.get("mbpp", {})) if evidence and isinstance(evidence.get("mbpp"), dict) else 0,
                         "full_review": False}
    return {
        "schema_version": 1, "assessment": "initial_screen_only", "status": "blocked",
        "full_review": False, "semantic_full_review": False,
        "baseline_assessment": {"source": f"{AUDIT}/PROGRESS.md", "status": "blocked",
                                 "known_anomalies_are_historical": True},
        "cache_reusable": reusable,
        "cache_reusable_scope": "prior static audit, raw source identities, and the named baseline candidate snapshot only; it does not cover a newer training pool",
        "audit_reports": reports,
        "cache": {"reusable": reusable, "issues": issues, "sources": sources,
                  "baseline_candidate_snapshot": candidate,
                  "missing_historical_snapshots": sorted(set(missing_history)),
                  "note": "This reuses only the old audit against the baseline candidate snapshot; a newer working pool needs its own overlap report. Missing historical snapshots do not invalidate this baseline match."},
        "evidence_coverage": evidence_coverage, "benchmarks": benchmarks,
        "integration_assessment": {
            "existing_components": [
                {"path": "scripts/freeze_lcb.py", "role": "freezes the LCB task manifest"},
                {"path": "scripts/freeze_evalplus.py", "role": "freezes task metadata, not candidate scores"},
                {"path": "src/posttrain/evaluation/extraction.py", "role": "extracts one Python program"},
                {"path": "src/posttrain/evaluation/statistics.py", "role": "summarizes paired outcomes"},
                {"path": "src/posttrain/execution/judge.py", "role": "scores stdio programs"},
                {"path": "src/posttrain/evaluation/lcb_results.py", "role": "adjudicates saved outputs; does not run a sandbox"},
            ],
            "minimum_next_steps": [
                "Accept the LCB checker through the isolated stdio judge using hash-bound synthetic results.",
                "Use one model-agnostic task loop to read frozen manifests, call the existing isolated executor, and persist per-task results.",
                "Pin the official EvalPlus loader/checker, execute functions in an isolated worker, and report base and plus separately.",
            ],
            "model_or_dataset_code_executed": False,
        },
    }
