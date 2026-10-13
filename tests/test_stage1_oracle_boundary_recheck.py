from scripts.stage1_oracle_boundary_recheck import (
    _field_target_occurrences,
    _labelled_review_targets,
    _target_occurrences_indexed,
    numeric_targets,
)


def test_numeric_targets_expand_powers_without_treating_decimal_parts_as_bounds():
    assert numeric_targets(["Add N=2*10^5 and values at 10^5 or 3.59*10^17."]) == [100000, 200000, 359000000000000000]


def test_target_literal_occurrence_binds_indices_and_keeps_field_mapping_unknown():
    tests = [
        {"stdin": "3 8\n", "stdout": "x\n"},
        {"stdin": "200000 4\n", "stdout": "y\n"},
        {"stdin": "8 200000\n", "stdout": "z\n"},
    ]
    evidence = _target_occurrences_indexed(tests, {0}, [[3, 8], [200000, 4], [8, 200000]], 200000)
    assert evidence["literal_token_present_anywhere"] is True
    assert evidence["literal_token_present_in_frozen_cases"] is False
    assert evidence["all_retained_test_indices_with_literal"] == [1, 2]
    assert evidence["field_binding_proven"] is False


def test_labelled_targets_and_contract_scan_separate_sample_omission_from_full_pool_absence():
    targets = _labelled_review_targets(["Add/review |S|=200000 and n=2*10^5."])
    assert [(item["field_name"], item["numeric_value_or_string_length"]) for item in targets] == [
        ("S", 200000), ("n", 200000)]
    contract = {
        "coverage": "complete",
        "operations": [{"id": "op0", "kind": "fields", "fields": [
            {"name": "n", "kind": "int", "bounds": [{"min_expr": {"constant": 1, "terms": {}},
                                                            "max_expr": {"constant": 200000, "terms": {}}}]}
        ]}],
    }
    tests = [{"stdin": "10\n", "stdout": "x\n"}, {"stdin": "200000\n", "stdout": "y\n"}]
    evidence = _field_target_occurrences(contract, tests, {0}, "n", 200000)
    assert evidence["field_binding_proven"] is True
    assert evidence["test_indices"] == [1]
    assert evidence["frozen_sample_test_indices"] == []
