import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrain.common import file_digest, write_json, write_jsonl
from posttrain.data.schema import Problem, TestCase as ProblemTestCase
from scripts import requalify_pending_retest as adapter


def _problem(problem_id: str) -> dict:
    return Problem(
        problem_id=problem_id,
        source="synthetic",
        revision="v1",
        source_id=problem_id,
        statement=f"Solve {problem_id}",
        tests=tuple(ProblemTestCase(stdin=f"{i}\n", stdout=f"{i}\n") for i in range(5)),
        references=(f"print({problem_id!r})", "print('alternate')"),
        cluster_id=f"cluster-{problem_id}",
        validation_status="pending_isolated_backtest",
    ).to_dict()


def _fixture(tmp_path: Path, monkeypatch, *, accepted_ids=("p1", "p3")):
    source_path = tmp_path / "cpu-backtest-source.jsonl"
    source_rows = [_problem(f"p{i}") for i in range(1, 5)]
    write_jsonl(source_path, source_rows)
    source_sha = file_digest(source_path)
    monkeypatch.setattr(adapter, "EXPECTED_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(adapter, "CANDIDATE_COUNT", len(source_rows))
    partitions = {
        "train": ["p1", "p2"],
        "dev": ["p3", "p4"],
        "calibration": ["p1"],
    }
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    partition_hashes = {}
    for name, ids in partitions.items():
        filename = f"{name}_ids.json"
        write_json(split_dir / filename, ids)
        partition_hashes[filename] = file_digest(split_dir / filename)
    split_report_path = split_dir / "report.json"
    write_json(split_report_path, {"synthetic": True})
    frozen = {
        "source_path": source_path,
        "source_sha256": source_sha,
        "split_report_path": split_report_path,
        "split_report_sha256": file_digest(split_report_path),
        "partition_ids": partitions,
        "official_eval_overlap_excluded": 0,
        "partition_files_sha256": partition_hashes,
        "split_dir": split_dir,
    }
    monkeypatch.setattr(adapter, "load_frozen_snapshot", lambda _root: frozen)

    run = tmp_path / "reference-run"
    run.mkdir()
    by_id = {row["problem_id"]: row for row in source_rows}
    shards = []
    for index, row in enumerate(source_rows):
        relative = f"inputs/shard-{index:02d}.jsonl"
        shard_path = run / relative
        write_jsonl(shard_path, [row])
        shards.append({
            "input": relative,
            "sha256": file_digest(shard_path),
            "problems": 1,
            "run": f"jobs/shard-{index:02d}",
        })
    write_json(run / "pool.json", {
        "source_sha256": source_sha,
        "input_problems": len(source_rows),
        "workers": 4,
        "max_references": 3,
        "shards": shards,
    })
    accepted_rows = [
        replace(Problem.from_dict(by_id[problem_id]),
                references=(Problem.from_dict(by_id[problem_id]).references[0],),
                validation_status="reference_passed").to_dict()
        for problem_id in accepted_ids
    ]
    write_jsonl(run / "accepted.jsonl", accepted_rows)
    write_json(run / "report.json", {"complete": True})
    report_sha = file_digest(run / "report.json")

    def verified(_run, supplied_sha, _scratch):
        assert supplied_sha == report_sha
        return {
            "all_checks_pass": True,
            "report_sha256": report_sha,
            "source_sha256": source_sha,
            "input_problems": len(source_rows),
            "reference_passed": len(accepted_rows),
            "quarantined": len(source_rows) - len(accepted_rows),
            "unknown_physical_attempts": 0,
            "physical_executions": 10,
            "bound_files_verified": 4,
        }

    monkeypatch.setattr(adapter, "verify_backup", verified)
    return tmp_path, run, source_path, partitions, report_sha


def _rewrite_shard(run: Path, problem_id: str, edit):
    pool = json.loads((run / "pool.json").read_text(encoding="utf-8"))
    for shard in pool["shards"]:
        path = run / shard["input"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if rows[0]["problem_id"] == problem_id:
            edit(rows[0])
            write_jsonl(path, rows)
            shard["sha256"] = file_digest(path)
            write_json(run / "pool.json", pool)
            write_json(run / "report.json", {
                "complete": True,
                "pool_manifest_sha256": file_digest(run / "pool.json"),
            })
            return
    raise AssertionError(f"synthetic shard missing {problem_id}")


def test_requalifies_only_passed_rows_without_changing_frozen_partitions(tmp_path, monkeypatch):
    root, run, _source_path, partitions, report_sha = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "qualified"

    result = adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result["train_reference_passed"] == 1
    assert result["dev_reference_passed"] == 1
    assert report["partition_ids"] == partitions
    assert report["calibration_reference_passed_ids"] == ["p1"]
    assert report["train_reference_passed_ids"] == ["p1"]
    assert report["dev_reference_passed_ids"] == ["p3"]
    assert report["quarantined_ids"] == ["p2", "p4"]
    assert report["training_ready"] is False
    assert report["public_eval_tasks_in_training"] is False
    assert report["partitions_changed"] is False
    outcomes = [json.loads(line) for line in
                (output / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert outcomes[0]["data_admission_status"] == "admitted_after_reference_backtest"
    assert outcomes[0]["data_admitted"] is True
    assert outcomes[1]["data_admitted"] is False
    assert report["train_data_admitted_count"] == 1
    assert report["training_readiness_note"] == "execution_environment_and_protocol_not_assessed_here"
    assert report["data_admitted_count"] == 2
    assert (output / "train-reference-passed.jsonl").is_file()
    dev_rows = [json.loads(line) for line in
                (output / "dev-reference-passed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["problem_id"] for row in dev_rows] == ["p3"]


def test_rejects_source_sha_drift_before_verifier_or_output(tmp_path, monkeypatch):
    root, run, _source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)
    pool = json.loads((run / "pool.json").read_text(encoding="utf-8"))
    pool["source_sha256"] = "f" * 64
    write_json(run / "pool.json", pool)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_unknown_or_incomplete_physical_execution(tmp_path, monkeypatch):
    root, run, source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)

    def unknown(*_args):
        return {
            "all_checks_pass": True,
            "report_sha256": report_sha,
            "source_sha256": file_digest(source_path),
            "input_problems": 4,
            "reference_passed": 2,
            "quarantined": 2,
            "unknown_physical_attempts": 1,
        }

    monkeypatch.setattr(adapter, "verify_backup", unknown)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="unknown_or_incomplete_physical_execution"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_mutated_accepted_problem_instead_of_admitting_it(tmp_path, monkeypatch):
    root, run, _source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)
    accepted = list(json.loads(line) for line in (run / "accepted.jsonl").read_text().splitlines())
    accepted[0]["statement"] = "altered after reference run"
    write_jsonl(run / "accepted.jsonl", accepted)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="accepted_problem_changed:p1"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_incomplete_result_coverage_and_duplicate_acceptance(tmp_path, monkeypatch):
    root, run, _source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)
    accepted = list(json.loads(line) for line in (run / "accepted.jsonl").read_text().splitlines())
    write_jsonl(run / "accepted.jsonl", [accepted[0], accepted[0]])
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="duplicate_accepted_problem_id"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_result_counts_that_do_not_match_verified_receipt(tmp_path, monkeypatch):
    root, run, source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "verify_backup", lambda *_: {
        "all_checks_pass": True,
        "report_sha256": report_sha,
        "source_sha256": file_digest(source_path),
        "input_problems": 4,
        "reference_passed": 3,
        "quarantined": 1,
        "unknown_physical_attempts": 0,
        "physical_executions": 10,
    })
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="verified_result_count_mismatch"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_missing_physical_execution_accounting(tmp_path, monkeypatch):
    root, run, source_path, _partitions, report_sha = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "verify_backup", lambda *_: {
        "all_checks_pass": True,
        "report_sha256": report_sha,
        "source_sha256": file_digest(source_path),
        "input_problems": 4,
        "reference_passed": 2,
        "quarantined": 2,
        "unknown_physical_attempts": 0,
        "physical_executions": None,
    })
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="unknown_or_incomplete_physical_execution"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


@pytest.mark.parametrize(
    ("problem_id", "edit", "error"),
    [
        ("p1", lambda row: row["tests"][0].update(stdout="drifted pass input"),
         "shard_problem_source_mismatch:p1"),
        ("p2", lambda row: row.update(references=["print('changed failed input')"]),
         "shard_problem_source_mismatch:p2"),
    ],
)
def test_rejects_shard_source_drift_for_passed_and_unpassed_rows(
        tmp_path, monkeypatch, problem_id, edit, error):
    root, run, _source, _partitions, _old_report_sha = _fixture(tmp_path, monkeypatch)
    _rewrite_shard(run, problem_id, edit)
    report_sha = file_digest(run / "report.json")
    monkeypatch.setattr(adapter, "verify_backup",
                        lambda *_: pytest.fail("source mismatch must fail before receipt verification"))
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match=error):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()


def test_rejects_reduced_reference_budget(tmp_path, monkeypatch):
    root, run, _source, _partitions, _old_report_sha = _fixture(tmp_path, monkeypatch)
    pool = json.loads((run / "pool.json").read_text(encoding="utf-8"))
    pool["max_references"] = 1
    write_json(run / "pool.json", pool)
    write_json(run / "report.json", {
        "complete": True,
        "pool_manifest_sha256": file_digest(run / "pool.json"),
    })
    report_sha = file_digest(run / "report.json")
    monkeypatch.setattr(adapter, "verify_backup",
                        lambda *_: pytest.fail("reference budget must fail before receipt verification"))
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="reference_budget_mismatch"):
        adapter.requalify(root, run, report_sha, output, tmp_path / "scratch")
    assert not output.exists()
