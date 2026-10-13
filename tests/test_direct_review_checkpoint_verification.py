"""Adversarial identity checks for the live direct-review checkpoints."""

import json

from posttrain.common import digest, file_digest, write_json
from scripts.verify_direct_review_checkpoints import text_sha, verify


def fixture_root(tmp_path):
    packet_root = tmp_path / ".cache/direct-review-shards-v2"
    packet_root.mkdir(parents=True)
    statement = "Return the input integer."
    test = {"stdin": "3\n", "stdout": "3\n"}
    packet = {"problem_id": "problem-1", "source_row_sha256": "bound-source-row",
              "statement_sha256": digest(statement),
              "tests": [test]}
    packet_path = packet_root / "review_01.jsonl"
    packet_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")
    manifest = {"source_sha256": "source", "excluded_problem_ids": [],
                "excluded_union_count": 0,
                "shards": [{"shard": 1, "path": packet_path.name,
                            "sha256": file_digest(packet_path),
                            "problem_ids": ["problem-1"]}]}
    write_json(packet_root / "manifest.json", manifest)
    checkpoint = tmp_path / "reports/direct_review_shard_01_checkpoint_v2.json"
    document = {"problems": [{"problem_id": "problem-1",
                               "statement_sha256": digest(statement),
                               "tests": [{"test_index": 0,
                                          "stdin_sha256": text_sha(test["stdin"]),
                                          "stdout_sha256": text_sha(test["stdout"]),
                                          "input_legality": "pass",
                                          "expected_output": "pass"}]}]}
    write_json(checkpoint, document)
    return tmp_path, checkpoint, document


def test_complete_direct_review_checkpoint(tmp_path):
    root, _, _ = fixture_root(tmp_path)
    result = verify(root)
    assert result["new_validated_review_count"] == 1
    assert result["new_validated_test_count"] == 1
    assert result["validated_source_row_sha256"]["problem-1"] == "bound-source-row"
    assert result["invalid_reviews"] == []
    assert result["semantic_verdicts_independently_proven"] is False


def test_changed_test_hash_cannot_be_counted(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    document["problems"][0]["tests"][0]["stdin_sha256"] = "0" * 64
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert result["invalid_reviews"][0]["issues"] == ["stdin_sha_mismatch"]


def test_one_based_test_index_cannot_be_counted(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    document["problems"][0]["tests"][0]["test_index"] = 1
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert "not_exact_zero_based_test_indices" in result["invalid_reviews"][0]["issues"]


def test_unknown_review_rows_cannot_be_silently_skipped(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    write_json(checkpoint, {"items": document["problems"], "reviewed_problem_count": 1})
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert result["invalid_reviews"][0]["issues"] == ["unrecognized_checkpoint_review_rows"]


def test_conflicting_output_verdict_fields_cannot_be_counted(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    test = document["problems"][0]["tests"][0]
    test["expected_output"] = {"status": "uncertain"}
    test["expected_output_verdict"] = "pass"
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert "conflicting_expected_output_verdicts" in result["invalid_reviews"][0]["issues"]


def test_literal_expected_stdout_can_coexist_with_verdict_alias(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    test = document["problems"][0]["tests"][0]
    test["expected_output"] = "3\n"
    test["expected_output_verdict"] = "pass"
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 1
    assert result["invalid_reviews"] == []


def test_malformed_structured_output_cannot_fall_back_to_alias(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    test = document["problems"][0]["tests"][0]
    test["expected_output"] = {"status": "bogus"}
    test["expected_output_verdict"] = "pass"
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert "conflicting_expected_output_verdicts" in result["invalid_reviews"][0]["issues"]


def test_inner_structured_output_conflict_cannot_be_counted(tmp_path):
    root, checkpoint, document = fixture_root(tmp_path)
    test = document["problems"][0]["tests"][0]
    test["expected_output"] = {"status": "pass", "verdict": "fail"}
    test["expected_output_verdict"] = "pass"
    write_json(checkpoint, document)
    result = verify(root)
    assert result["new_validated_review_count"] == 0
    assert "conflicting_expected_output_verdicts" in result["invalid_reviews"][0]["issues"]
