"""Synthetic tests for the frozen pre-backtest TACO split."""
import importlib.util
import json
from pathlib import Path

import pytest

from posttrain.common import digest, file_digest, read_jsonl, write_json, write_jsonl
from posttrain.data.schema import Problem, TestCase as Case


def _problem(index, *, source_id=None, cluster_id=None, statement=None):
    return Problem(
        problem_id=f"taco-train-{index:06d}",
        source="BAAI/TACO",
        revision="synthetic-fixed-revision",
        source_id=source_id or f"https://example.test/task/{index}",
        statement=statement or f"alpha{index} beta{index} gamma{index} delta{index} epsilon{index}",
        tests=tuple(Case(f"input-{index}-{j}", f"output-{index}-{j}") for j in range(5)),
        references=("reference-a", "reference-b", "reference-c"),
        difficulty=("EASY", "MEDIUM", "HARD")[index % 3],
        cluster_id=cluster_id or f"parent-cluster-{index}",
        validation_status="unverified",
    )


def _public_files(root, candidate_statement=None):
    manifests = root / "manifests"
    manifests.mkdir(parents=True)

    lcb_rows = []
    lcb_tasks = []
    for i in range(200):
        statement = candidate_statement if i == 0 and candidate_statement else (
            f"lcbalpha{i} lcbbeta{i} lcbgamma{i} lcbdelta{i} lcbepsilon{i}")
        task_id = f"lcb-task-{i:03d}"
        lcb_rows.append({"task_id": task_id, "statement": statement})
        lcb_tasks.append({"task_id": task_id, "source_id": f"lcb://task/{i}",
                          "statement_hash": digest(statement)})
    write_jsonl(root / "lcb-metadata.jsonl", lcb_rows)
    write_json(manifests / "lcb_stdio_200.json", {"tasks": lcb_tasks})

    eval_rows = []
    humaneval = []
    mbpp = []
    for i in range(542):
        task_id = f"evalplus-task-{i:03d}"
        statement = f"evalalpha{i} evalbeta{i} evalgamma{i} evaldelta{i} evalepsilon{i}"
        eval_rows.append({"task_id": task_id, "source_id": f"evalplus://task/{i}",
                          "statement": statement})
        task = {"task_id": task_id, "prompt_hash": digest(statement)}
        (humaneval if i < 164 else mbpp).append(task)
    write_jsonl(root / "evalplus-public.jsonl", eval_rows)
    write_json(manifests / "humaneval_plus.json", {"tasks": humaneval})
    write_json(manifests / "mbpp_plus.json", {"tasks": mbpp})
    return manifests


def _bundle(tmp_path, *, count=260, final_overlap=False):
    input_dir = tmp_path / "pending"
    parent_dir = tmp_path / "parent"
    input_dir.mkdir()
    parent_dir.mkdir()

    parent = []
    candidates = []
    lineage = []
    for i in range(count):
        source_id = f"https://example.test/task/{i}"
        cluster_id = "parent-bridge-cluster" if i < 2 else None
        p = _problem(i, source_id=source_id, cluster_id=cluster_id)
        parent.append(p.to_dict())
        candidate = p.to_dict()
        candidate["metadata"]["cleaning_parent_tests_hash"] = digest(p.to_dict()["tests"])
        if i == 0:
            candidate["references"] = [p.references[1]]
        candidate["validation_status"] = "pending_isolated_backtest"
        candidates.append(candidate)
        lineage.append({"problem_id": p.problem_id, "cluster_id": p.cluster_id,
                        "source_row_sha256": digest(p.to_dict()),
                        "source_statement_sha256": digest(p.statement),
                        "source_tests_sha256": digest(candidate["tests"]),
                        "retained_source_test_indices": list(range(len(candidate["tests"]))),
                        "retained_test_hashes": [digest(test) for test in candidate["tests"]],
                        "effective_unique_test_count": len(candidate["tests"]),
                        "included": True, "exclusion_reasons": []})

    excluded = []
    unfinished = []
    for i in range(61):
        held = _problem(10000 + i, cluster_id=f"held-cluster-{i}").to_dict()
        held["problem_id"] = f"taco-train-unfinished-{i:03d}"
        parent.append(held)
        row = {"problem_id": held["problem_id"],
               "cluster_id": held["cluster_id"],
               "source_row_sha256": digest(held),
               "source_statement_sha256": digest(held["statement"]),
               "source_tests_sha256": digest(held["tests"]),
               "retained_source_test_indices": list(range(len(held["tests"]))),
               "retained_test_hashes": [], "effective_unique_test_count": 0,
               "included": False, "exclusion_reasons": ["unfinished_direct_review"]}
        lineage.append(row)
        excluded.append({"problem_id": row["problem_id"], "source_row_sha256": row["source_row_sha256"],
                         "statement_sha256": row["source_statement_sha256"],
                         "shard": i % 4, "partition": "unfinished"})
        unfinished.append(row["problem_id"])
    held = _problem(11000, cluster_id="held-confirmed").to_dict()
    held["problem_id"] = "taco-train-confirmed-block-000"
    parent.append(held)
    lineage.append({"problem_id": held["problem_id"],
                    "cluster_id": held["cluster_id"],
                    "source_row_sha256": digest(held),
                    "source_statement_sha256": digest(held["statement"]),
                    "source_tests_sha256": digest(held["tests"]),
                    "retained_source_test_indices": list(range(len(held["tests"]))),
                    "retained_test_hashes": [], "effective_unique_test_count": 0,
                    "included": False, "exclusion_reasons": ["confirmed_invalid_test_input"]})

    write_jsonl(input_dir / "problems.jsonl", candidates)
    write_jsonl(input_dir / "lineage.jsonl", lineage)
    write_jsonl(input_dir / "unfinished_direct_review.jsonl", excluded)
    finalizer_report = {
        "scope": "reviewed_static_pool_pending_isolated_backtest",
        "status": "pending_isolated_backtest",
        "hidden_final_tests_loaded": False,
        "unfinished_direct_review_ids": sorted(unfinished),
        "unfinished_direct_review_ids_sha256": digest(sorted(unfinished)),
        "confirmed_hold_ids": ["taco-train-confirmed-block-000"],
        "confirmed_hold_ids_sha256": digest(["taco-train-confirmed-block-000"]),
        "confirmed_hold_count": 1,
        "counts": {"source_problems": len(lineage), "candidate_problems": count,
                   "unfinished_direct_review": 61, "excluded_problems": len(lineage) - count},
        "fresh_isolated_executions": 0,
        "fresh_backtest_complete": False,
        "pending_isolated_backtest": True,
        "dedup_split_complete": False,
        "human_review_complete": False,
        "semantic_verdicts_independently_proven": False,
        "training_ready": False,
    }
    write_jsonl(input_dir / "exclusions.jsonl", [row for row in lineage if not row["included"]])
    write_jsonl(input_dir / "pending.jsonl", [
        {"problem_id": row["problem_id"],
         "status": "pending_isolated_backtest",
         "source_row_sha256": row["source_row_sha256"],
         "effective_unique_test_count": row["effective_unique_test_count"],
         "training_ready": False}
        for row in lineage if row["included"]
    ])

    write_jsonl(parent_dir / "problems.jsonl", parent)
    write_jsonl(parent_dir / "dedup_links.jsonl", [
        {"left": parent[1]["problem_id"], "right": parent[0]["problem_id"],
         "reason": "source", "similarity": 1.0}])
    manifests = _public_files(tmp_path / "public", parent[0]["statement"] if final_overlap else None)
    public = tmp_path / "public"
    parent_report = {
        "candidate_manifest_sha256": file_digest(parent_dir / "problems.jsonl"),
        "retained": len(parent),
        "clusters": len({row["cluster_id"] for row in parent}),
        "dedup_links": 1,
        "final_public_tasks": 742,
        "final_manifests": {p.name: file_digest(p) for p in manifests.iterdir()},
        "removed_final_overlap_ids": [],
        "hidden_final_tests_loaded": False,
        "training_ready": False,
    }
    if final_overlap:
        # The parent report binds the previous frozen public set, so a changed
        # manifest must force a fresh public decontamination pass.
        parent_report["final_manifests"]["lcb_stdio_200.json"] = "0" * 64
    write_json(parent_dir / "report.json", parent_report)
    finalizer_report["identities"] = {
        "source_path": "data/processed/taco-training-pool-v4/problems.jsonl",
        "source_sha256": file_digest(parent_dir / "problems.jsonl"),
        "input_sha256": {"data/processed/taco-training-pool-v4/problems.jsonl":
                         file_digest(parent_dir / "problems.jsonl")},
    }
    finalizer_report["artifacts"] = {name: file_digest(input_dir / name) for name in (
        "problems.jsonl", "lineage.jsonl", "exclusions.jsonl",
        "unfinished_direct_review.jsonl", "pending.jsonl")}
    write_json(input_dir / "report.json", finalizer_report)
    return input_dir, parent_dir, manifests, public / "evalplus-public.jsonl", public / "lcb-metadata.jsonl"


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts/split_taco_pending_retest.py"
    spec = importlib.util.spec_from_file_location("split_taco_pending_retest", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prebacktest_split_keeps_clusters_together_and_calibration_in_train(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    output = tmp_path / "split"

    result = _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    source = list(read_jsonl(output / "cpu-backtest-source.jsonl"))
    train = json.loads((output / "train_ids.json").read_text(encoding="utf-8"))
    dev = json.loads((output / "dev_ids.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "calibration_ids.json").read_text(encoding="utf-8"))
    cluster_by_id = {row["problem_id"]: row["cluster_id"] for row in source}
    assert len(source) == 260
    assert set(train).isdisjoint(dev)
    assert set(train) | set(dev) == set(cluster_by_id)
    assert {cluster_by_id[i] for i in train}.isdisjoint({cluster_by_id[i] for i in dev})
    assert cluster_by_id["taco-train-000000"] == cluster_by_id["taco-train-000001"]
    assert result["source_cluster_overlap"] == 0
    assert len(calibration) == 200
    assert set(calibration) <= set(train)
    assert all(row["validation_status"] == "pending_isolated_backtest" for row in source)
    assert result["status"] == "pending_isolated_backtest"
    assert result["training_ready"] is False
    assert result["seed"] == 20260925
    assert result["dev_fraction"] == 0.15
    assert result["reference_count_max"] == 3
    assert next(row for row in source if row["problem_id"] == "taco-train-000000")["references"] == ["reference-b"]

    reasons = {row["problem_id"]: row["exclusion_reasons"]
               for row in read_jsonl(output / "exclusions.jsonl")}
    assert reasons["taco-train-unfinished-000"] == ["unfinished_direct_review"]
    assert reasons["taco-train-confirmed-block-000"] == ["confirmed_invalid_test_input"]


def test_finalizer_input_hash_drift_fails_before_creating_output(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    rows = list(read_jsonl(input_dir / "problems.jsonl"))
    rows[0]["statement"] += " changed"
    write_jsonl(input_dir / "problems.jsonl", rows)
    output = tmp_path / "split"

    with pytest.raises(ValueError):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_duplicate_candidate_ids_fail_closed_even_when_finalizer_rebinds_file(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    rows = list(read_jsonl(input_dir / "problems.jsonl"))
    rows[-1] = rows[0]
    write_jsonl(input_dir / "problems.jsonl", rows)
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    report["artifacts"]["problems.jsonl"] = file_digest(input_dir / "problems.jsonl")
    write_json(input_dir / "report.json", report)
    output = tmp_path / "split"

    with pytest.raises(ValueError):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_candidate_tests_must_match_retained_source_indices_and_lineage_hash(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    rows = list(read_jsonl(input_dir / "problems.jsonl"))
    rows[0]["tests"][0]["stdout"] = "tampered-output"
    write_jsonl(input_dir / "problems.jsonl", rows)
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    report["artifacts"]["problems.jsonl"] = file_digest(input_dir / "problems.jsonl")
    write_json(input_dir / "report.json", report)
    output = tmp_path / "split"

    with pytest.raises(ValueError):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_candidate_reference_set_must_be_preserved_from_source(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    rows = list(read_jsonl(input_dir / "problems.jsonl"))
    rows[0]["references"] = ["print(0)"]
    write_jsonl(input_dir / "problems.jsonl", rows)
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    report["artifacts"]["problems.jsonl"] = file_digest(input_dir / "problems.jsonl")
    write_json(input_dir / "report.json", report)
    output = tmp_path / "split"

    with pytest.raises(ValueError):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_candidate_non_source_fields_must_remain_identical(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    rows = list(read_jsonl(input_dir / "problems.jsonl"))
    rows[0]["difficulty"] = "HARD"
    write_jsonl(input_dir / "problems.jsonl", rows)
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    report["artifacts"]["problems.jsonl"] = file_digest(input_dir / "problems.jsonl")
    write_json(input_dir / "report.json", report)
    output = tmp_path / "split"

    with pytest.raises(ValueError, match="candidate_non_source_field_changed:difficulty"):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_candidate_source_row_hash_must_match_v4_parent(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path)
    lineage = list(read_jsonl(input_dir / "lineage.jsonl"))
    lineage[0]["source_row_sha256"] = "0" * 64
    write_jsonl(input_dir / "lineage.jsonl", lineage)
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    report["artifacts"]["lineage.jsonl"] = file_digest(input_dir / "lineage.jsonl")
    write_json(input_dir / "report.json", report)
    output = tmp_path / "split"

    with pytest.raises(ValueError):
        _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    assert not output.exists()


def test_changed_frozen_final_manifest_recomputes_and_excludes_full_overlap_cluster(tmp_path):
    input_dir, parent_dir, manifests, evalplus, lcb = _bundle(tmp_path, final_overlap=True)
    output = tmp_path / "split"

    result = _module().allocate(input_dir, parent_dir, manifests, evalplus, lcb, output)

    source_ids = {row["problem_id"] for row in read_jsonl(output / "cpu-backtest-source.jsonl")}
    excluded = {row["problem_id"]: row["exclusion_reasons"]
                for row in read_jsonl(output / "exclusions.jsonl")}
    assert "taco-train-000000" not in source_ids
    assert "taco-train-000001" not in source_ids
    assert "official_final_eval_overlap" in excluded["taco-train-000000"]
    assert "official_final_eval_overlap" in excluded["taco-train-000001"]
    assert result["final_overlap_method"] == "joint_public_recompute"
