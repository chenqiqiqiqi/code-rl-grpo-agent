from posttrain.data.constraint_engine import compile_contract, validate_input


def test_scalar_bounds_are_bound_to_the_input_field_and_include_exact_evidence():
    statement = """Input
The first line contains one integer n (1 <= n <= 4).
Output
Print an integer between 1 and 2.
Examples
Input
99
Output
1
"""

    contract = compile_contract(statement)
    assert contract["coverage"] == "complete"
    assert validate_input(contract, "4\n")["violations"] == []

    result = validate_input(contract, "5\n")
    assert len(result["violations"]) == 1
    violation = result["violations"][0]
    assert violation["reason"] == "integer_out_of_range"
    evidence = violation["evidence"]
    assert evidence and all(statement[item["start"] : item["end"]] == item["text"] for item in evidence)
    assert any("1 <= n <= 4" in item["text"] for item in evidence)


def test_counted_integer_array_uses_the_declared_length_and_element_bounds():
    statement = """Input
The first line contains one integer n (1 <= n <= 3).
The second line contains n integers a_1, a_2, ..., a_n (0 <= a_i <= 5).
Output
Print the array sum.
"""

    contract = compile_contract(statement)
    assert contract["coverage"] == "complete"
    assert validate_input(contract, "3\n0 4 5\n")["violations"] == []
    result = validate_input(contract, "2\n0 6\n")
    assert [item["reason"] for item in result["violations"]] == ["integer_out_of_range"]
    assert result["violations"][0]["token_index"] == 2


def test_explicit_cross_variable_constraint_is_checked_without_evaluating_source():
    statement = """Input
The first line contains two integers x and y (1 <= x, y <= 10^5, x + y <= 10^5).
Output
Print x + y.
"""

    contract = compile_contract(statement)
    assert validate_input(contract, "50000 50000")["violations"] == []
    result = validate_input(contract, "50001 50000")
    assert any(item["reason"] == "constraint_violated" for item in result["violations"])


def test_flattened_exponent_like_109_does_not_become_a_bound():
    statement = """Input
The first line contains one integer n (1 <= n <= 109).
Output
Print n.
"""

    contract = compile_contract(statement)
    assert contract["coverage"] == "partial"
    assert validate_input(contract, "110")["violations"] == []


def test_bounds_from_output_and_examples_are_never_applied_to_input():
    statement = """Input
The first line contains one integer n.
Output
The answer is between 1 and 3.
Examples
Input
100
Output
3
"""

    contract = compile_contract(statement)
    assert contract["coverage"] == "partial"
    assert validate_input(contract, "100")["violations"] == []


def test_only_explicit_permutation_semantics_enable_duplicate_check():
    statement = """Input
The first line contains one integer n (1 <= n <= 4).
The second line contains n integers p_1, ..., p_n. The array p is a permutation of 1, ..., n.
Output
Print the permutation.
"""

    contract = compile_contract(statement)
    assert validate_input(contract, "3\n1 1 3")["violations"][0]["reason"] == "not_a_permutation"
    assert validate_input(contract, "3\n3 1 2")["violations"] == []


def test_explicit_tree_with_n_minus_one_edges_rejects_a_cycle_or_duplicate_edge():
    statement = """Input
The first line contains one integer n (2 <= n <= 4).
Each of the next n - 1 lines contains two integers u_i and v_i (1 <= u_i, v_i <= n).
The given graph is a tree.
Output
Print the graph.
"""

    contract = compile_contract(statement)
    result = validate_input(contract, "3\n1 2\n1 2\n")
    assert any(item["reason"] == "not_a_tree" for item in result["violations"])


def test_missing_or_ambiguous_input_grammar_is_unsupported_without_rejection():
    contract = compile_contract("Solve the task and print the answer.")
    assert contract["coverage"] == "unsupported"
    result = validate_input(contract, "anything at all")
    assert result["coverage"] == "unsupported"
    assert result["violations"] == []


def test_grouped_numeric_bounds_are_not_truncated_to_the_first_group():
    for bound, accepted, rejected in [
        ("1 000 000", "2", "1000001"),
        ("100,000", "101", "100001"),
        ("10^{18}", "11", "1000000000000000001"),
        ("2*10^5", "199999", "200001"),
    ]:
        statement = f"""Input
The first line contains one integer n (1 <= n <= {bound}).
Output
Print n.
"""
        contract = compile_contract(statement)
        assert validate_input(contract, accepted)["violations"] == []
        assert any(item["reason"] == "integer_out_of_range" for item in validate_input(contract, rejected)["violations"])


def test_unparsed_intermediate_line_stops_later_positional_binding():
    statement = """Input
The first line contains one integer n (1 <= n <= 3).
The second line consists of a specially encoded value.
The third line contains one integer m (1 <= m <= 3).
Output
Print m.
"""

    contract = compile_contract(statement)
    result = validate_input(contract, "1\nnot-an-integer\n2\n")
    assert result["coverage"] == "partial"
    assert not any(item["reason"] == "expected_integer" for item in result["violations"])


def test_very_long_lexical_integer_is_not_misreported_as_a_non_integer():
    contract = compile_contract("""Input
The first line contains one integer n.
Output
Print n.
""")

    result = validate_input(contract, "9" * 5000)
    assert result["coverage"] == "partial"
    assert result["violations"] == []


def test_reviewed_complete_layout_does_not_upgrade_partial_semantics():
    statement = "Input\nThe first line contains one integer n (1 <= n <= 3).\nThe second line contains n integers a_1, a_2, ..., a_n (1 <= a_i <= 5).\nOutput\nPrint the sum."
    contract = compile_contract(statement)
    contract.update(coverage="partial", limitations=["distinctness_unchecked"], layout_complete=True)
    assert validate_input(contract, "2 1 2")["coverage"] == "partial"
    assert validate_input(contract, "2 1 2")["violations"] == []
    assert any(v["reason"] == "trailing_input_tokens" for v in validate_input(contract, "2 1 2 3")["violations"])
    assert any(v["reason"] == "missing_input_token" for v in validate_input(contract, "2 1")["violations"])
    contract.pop("layout_complete")
    assert validate_input(contract, "2 1 2 3")["violations"] == []
