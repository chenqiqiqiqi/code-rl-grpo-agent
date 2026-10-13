import hashlib
import importlib
import json
from pathlib import Path

import pytest


package = importlib.import_module("scripts.package_pending_retest")


def row(problem_id, cluster_id, status="pending_isolated_backtest"):
    return {
        "problem_id": problem_id,
        "cluster_id": cluster_id,
        "validation_status": status,
    }


def valid_partition():
    candidates = [row("p1", "c1"), row("p2", "c2"), row("p3", "c3")]
    return {
        "candidate_rows": candidates,
        "backtest_rows": [dict(item) for item in candidates],
        "train_ids": ["p1", "p2"],
        "dev_ids": ["p3"],
        "calibration_ids": ["p2"],
        "cluster_by_id": {"p1": "c1", "p2": "c2", "p3": "c3"},
        "excluded_ids": {"held"},
        "unfinished_ids": {"unfinished"},
        "reported_cluster_overlap": 0,
    }


def test_training_snapshot_accepts_exact_pending_disjoint_partition():
    result = package.validate_training_partition(**valid_partition())

    assert result == {
        "candidate_count": 3,
        "train_count": 2,
        "dev_count": 1,
        "calibration_count": 1,
        "cluster_overlap": 0,
        "training_ready": False,
        "status": "pending_isolated_backtest",
    }


@pytest.mark.parametrize("problem_id", ["held", "unfinished"])
def test_training_snapshot_rejects_hold_or_unfinished_direct_review(problem_id):
    snapshot = valid_partition()
    snapshot["candidate_rows"][0]["problem_id"] = problem_id
    snapshot["backtest_rows"][0]["problem_id"] = problem_id
    snapshot["train_ids"][0] = problem_id
    snapshot["cluster_by_id"][problem_id] = "c1"

    with pytest.raises(ValueError, match="excluded_problem_in_training_pool"):
        package.validate_training_partition(**snapshot)


def test_training_snapshot_rejects_cross_split_cluster_and_count_drift():
    snapshot = valid_partition()
    snapshot["cluster_by_id"]["p3"] = "c1"
    snapshot["candidate_rows"][2]["cluster_id"] = "c1"
    snapshot["backtest_rows"][2]["cluster_id"] = "c1"
    with pytest.raises(ValueError, match="cluster_overlap"):
        package.validate_training_partition(**snapshot)

    snapshot = valid_partition()
    snapshot["backtest_rows"].pop()
    with pytest.raises(ValueError, match="source_count_or_membership_mismatch"):
        package.validate_training_partition(**snapshot)


def test_registered_training_artifact_hashes_fail_closed_on_missing_or_drift(tmp_path):
    artifact = tmp_path / "problems.jsonl"
    artifact.write_bytes(b'{"problem_id":"p1"}\n')
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert package.verify_artifact_hashes(tmp_path, {"problems.jsonl": expected}) == {
        "problems.jsonl": expected
    }

    artifact.write_bytes(b'{"problem_id":"tampered"}\n')
    with pytest.raises(ValueError, match="artifact_hash_mismatch:problems.jsonl"):
        package.verify_artifact_hashes(tmp_path, {"problems.jsonl": expected})

    artifact.unlink()
    with pytest.raises(ValueError, match="artifact_missing:problems.jsonl"):
        package.verify_artifact_hashes(tmp_path, {"problems.jsonl": expected})


def test_delivery_verifier_detects_readme_tampering(tmp_path):
    source = tmp_path / "data" / "source.jsonl"
    source.parent.mkdir()
    source.write_bytes(b"source\n")
    output = tmp_path / "reports" / "pending-retest-delivery"
    output.mkdir(parents=True)
    manifest = {
        "training": {"status": "pending_isolated_backtest", "files": [
            {"path": "data/source.jsonl", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ]},
        "evaluation": {"status": "blocked", "files": []},
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "README.md").write_text("source-bound summary\n", encoding="utf-8")
    expected = {
        "data/source.jsonl": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reports/pending-retest-delivery/MANIFEST.json": hashlib.sha256(
            (output / "MANIFEST.json").read_bytes()).hexdigest(),
        "reports/pending-retest-delivery/README.md": hashlib.sha256(
            (output / "README.md").read_bytes()).hexdigest(),
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(expected.items())),
        encoding="utf-8",
    )

    assert package.verify_delivery(tmp_path, Path("reports/pending-retest-delivery"))["verified"]
    (output / "README.md").write_text("changed summary\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_list_entry_mismatch:reports/pending-retest-delivery/README.md"):
        package.verify_delivery(tmp_path, Path("reports/pending-retest-delivery"))
