import hashlib
import pytest

from scripts.stage1_oracle_coverage import (
    _case_hashes,
    _simple_boundary_diagnostic,
    build_stratum_coverage,
    classify_input_format,
    deduplicate_tests,
    selected_case_evidence,
    make_review_samples,
)


def _row(pid, statement, tests, *, difficulty="EASY", cluster=None):
    return {
        "problem_id": pid,
        "cluster_id": cluster or f"cluster-{pid}",
        "difficulty": difficulty,
        "statement": statement,
        "tests": [{"stdin": stdin, "stdout": stdout} for stdin, stdout in tests],
    }


def test_dedup_diagnostic_distinguishes_duplicate_pairs_and_conflicting_oracles():
    result = deduplicate_tests([
        {"stdin": "1\r\n", "stdout": "2\n"},
        {"stdin": "1\n", "stdout": "2  \n"},
        {"stdin": "1\n", "stdout": "3\n"},
        {"stdin": "1  2\n", "stdout": "2\n"},
    ])

    assert result["retained_test_count"] == 4
    assert result["unique_test_pair_count"] == 3
    assert result["duplicate_test_pair_excess_count"] == 1
    assert result["same_input_conflict_group_count"] == 1
    assert result["same_input_conflict_groups"][0]["test_indices"] == [0, 1, 2]


def test_input_format_is_explicitly_heuristic_and_uses_statement_context():
    scalar = _row("a", "Read one integer n.", [("1\n", "1"), ("9\n", "9")])
    graph = _row("b", "The input describes a tree with N vertices and N-1 edges.",
                 [("2\n1 2\n", "1"), ("3\n1 2\n2 3\n", "1")])

    assert classify_input_format(scalar) == "single_token"
    assert classify_input_format(graph) == "graph_or_tree"


def test_scalar_boundary_diagnostic_reports_missing_declared_upper_bound():
    row = _row("a", "Read x with 1 <= x <= 10.", [("1\n", "0"), ("4\n", "1")])
    audit = {"contract": {"kind": "integer", "min": 1, "max": 10}}

    result = _simple_boundary_diagnostic(row, audit)

    assert result["status"] == "direct_scalar_range_diagnostic"
    assert result["declared_min_test_indices"] == [0]
    assert result["declared_max_test_indices"] == []
    assert result["observed_max"] == 4


def test_fixed_stratification_selects_one_candidate_per_nonempty_cell():
    first = _row("a", "Read one integer x.", [("1", "1")], difficulty="EASY")
    second = _row("b", "Read one integer x.", [("2", "2")], difficulty="EASY")
    third = _row("c", "Read an array of n integers.", [("2\n1 2\n", "2")], difficulty="HARD")
    audits = {
        "a": {"contract": {"kind": "integer", "min": 1, "max": 10}, "input_coverage": "complete"},
        "b": {"contract": {"kind": "integer", "min": 1, "max": 10}, "input_coverage": "complete"},
        "c": {"contract": {"compiler": "manual", "constraints": [], "operations": []}, "input_coverage": "complete"},
    }

    first_run = make_review_samples([first, second, third], audits)
    second_run = make_review_samples([first, second, third], audits)

    assert [item["problem_id"] for item in first_run] == [item["problem_id"] for item in second_run]
    assert len(first_run) == 2
    assert {item["difficulty"] for item in first_run} == {"EASY", "HARD"}
    assert all(item["statement_utf8_raw_bytes_sha256"] for item in first_run)
    assert all(item["selected_cases"][0]["test_pair_canonical_json_sha256"] for item in first_run)


def test_current_stratum_report_distinguishes_individual_unknown_from_unrepresented_cell():
    statement = "There are t test cases. For each test case, read two integers."
    rows = [
        _row("sampled", statement, [("1 2\n", "3")], difficulty="EASY"),
        _row("unreviewed", statement, [("2 1\n", "3")], difficulty="EASY"),
        _row("unrepresented", "Read one integer.", [("7\n", "7")], difficulty="EASY"),
    ]
    audits = {
        row["problem_id"]: {
            "contract": {"operations": ["one row"], "constraints": [{"kind": "range"}]},
        }
        for row in rows[:2]
    }
    audits["unrepresented"] = {"contract": {"kind": "integer", "min": 1, "max": 10}}

    result = build_stratum_coverage(rows, audits, {"sampled"})

    assert result["reviewed_strata_counts"] == {
        "EASY|query_or_multi_case|multi_case_structured_ranges": 1,
    }
    assert result["unrepresented_strata"] == [{
        "stratum_key": "EASY|single_token|scalar_integer",
        "candidate_count": 1,
        "problem_ids": ["unrepresented"],
    }]
    assert result["candidate_review_status_by_id"]["unreviewed"] == (
        "not_individually_reviewed_in_represented_stratum_unknown"
    )
    assert result["candidate_review_status_by_id"]["unrepresented"] == "stratum_not_sampled_unknown"


def test_selected_case_evidence_binds_case_index_input_hash_and_stdout_repr():
    row = _row("taco-train-000659", "Read a string.", [("aba\n", "Second\n"), ("wewe\n", "Second\n")])
    selected = {**_case_hashes(row["tests"][1], 1), "selection_reason": "longest_input_bytes"}
    sample = {"selected_cases": [selected]}

    evidence = selected_case_evidence(row["problem_id"], row, sample)
    input_hash = hashlib.sha256("wewe\n".encode("utf-8")).hexdigest()
    output_hash = hashlib.sha256("Second\n".encode("utf-8")).hexdigest()

    assert evidence[0]["case_id"] == f"taco-train-000659:case:1:{input_hash}"
    assert evidence[0]["case_index"] == 1
    assert evidence[0]["stdin_utf8_raw_bytes_sha256"] == input_hash
    assert evidence[0]["stdin_length_utf8_bytes"] == len("wewe\n".encode("utf-8"))
    assert evidence[0]["stdin_repr_summary"] == "'wewe\\n'"
    assert evidence[0]["stdout_utf8_raw_bytes_sha256"] == output_hash
    assert evidence[0]["stdout_length_utf8_bytes"] == len("Second\n".encode("utf-8"))
    assert evidence[0]["stdout_repr_summary"] == "'Second\\n'"

    tampered_sample = {"selected_cases": [{**selected, "stdout_utf8_raw_bytes_sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="selected_case_hash_mismatch:taco-train-000659:1"):
        selected_case_evidence(row["problem_id"], row, tampered_sample)
