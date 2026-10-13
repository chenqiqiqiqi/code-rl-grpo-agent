from posttrain.data.statement_quality import assess_statement


def test_example_only_statement_is_quarantined_with_exact_evidence():
    result = assess_statement("Example")

    assert result["status"] == "quarantine"
    assert result["issues"] == [
        {"reason": "example_only_statement", "evidence": "Example", "start": 0, "end": 7}
    ]


def test_input_output_format_without_a_task_mapping_is_quarantined():
    statement = (
        "-----Input-----\n\n"
        "The input contains a single integer $a$ ($1 \\le a \\le 99$).\n\n"
        "-----Output-----\n\n"
        'Output "YES" or "NO".\n\n'
        "-----Examples-----\nInput\n5\nOutput\nYES"
    )

    result = assess_statement(statement)

    assert result["status"] == "quarantine"
    assert result["issues"] == [
        {
            "reason": "missing_task_definition",
            "evidence": 'Output "YES" or "NO".',
            "start": statement.index('Output "YES" or "NO"'),
            "end": statement.index('Output "YES" or "NO"') + len('Output "YES" or "NO".'),
        }
    ]


def test_bare_scalar_input_output_schema_is_missing_a_task_mapping():
    result = assess_statement(
        "Input\nThe input contains a single integer a (1 <= a <= 64).\n"
        "Output\nOutput a single integer."
    )

    assert result["status"] == "quarantine"
    assert result["issues"][0]["reason"] == "missing_task_definition"


def test_bare_two_integer_yes_no_schema_is_missing_a_task_mapping():
    result = assess_statement(
        "Input\nThe input contains two integers N, M (1 <= N <= 1024, 2 <= M <= 16), "
        "separated by a single space.\nOutput\nOutput YES or NO."
    )

    assert result["status"] == "quarantine"
    assert result["issues"][0]["reason"] == "missing_task_definition"


def test_compact_input_output_formula_is_a_complete_task_definition():
    statement = "Input: one integer n. Output n modulo 2."

    result = assess_statement(statement)

    assert result == {"status": "no_confirmed_issue", "issues": []}


def test_example_sample_data_is_not_used_as_a_task_definition():
    result = assess_statement("Example\nInput\n3\nOutput\n1")

    assert result["status"] == "quarantine"
    assert result["issues"][0]["reason"] == "example_only_statement"


def test_task_like_prose_after_a_leading_example_heading_needs_review():
    result = assess_statement("Example\nInput\n3\nOutput\n1\nFind the number of valid strings.")

    assert result["status"] == "review_needed"
    assert result["issues"][0]["reason"] == "task_text_after_example_heading"


def test_sixteen_character_target_is_not_reported_as_a_length_contradiction():
    statement = (
        "Change S to CODEFESTIVAL2016. S is 16 characters long. "
        "Examples: C0DEFESTIVAL2O16 and FESTIVAL2016CODE"
    )

    result = assess_statement(statement)

    assert len("CODEFESTIVAL2016") == 16
    assert len("C0DEFESTIVAL2O16") == 16
    assert len("FESTIVAL2016CODE") == 16
    assert result == {"status": "no_confirmed_issue", "issues": []}


def test_impossible_chained_numeric_constraint_is_quarantined_with_exact_span():
    statement = "Find x.\nConstraints\n1 <= x <= 0"

    result = assess_statement(statement)

    start = statement.index("1 <= x <= 0")
    assert result["status"] == "quarantine"
    assert result["issues"] == [
        {
            "reason": "contradictory_numeric_bounds",
            "evidence": "1 <= x <= 0",
            "start": start,
            "end": start + len("1 <= x <= 0"),
        }
    ]


def test_unicode_and_latex_impossible_bounds_are_quarantined():
    for statement in ("Constraints\n3 ≤ n ≤ 2", "Constraints\n3 \\leq n \\leq 2"):
        result = assess_statement(statement)

        assert result["status"] == "quarantine"
        assert result["issues"][0]["reason"] == "contradictory_numeric_bounds"


def test_impossible_numeric_condition_in_narrative_is_not_quarantined():
    result = assess_statement("If 1 <= x <= 0, no solution exists; print -1.")

    assert result["status"] == "review_needed"
    assert result["issues"][0]["reason"] == "contradictory_numeric_bounds_contextual"


def test_mixed_direction_bound_chains_do_not_invent_a_contradiction():
    for statement in (
        "Constraints\n1 <= x >= 0",
        "Constraints\n5 >= x <= 7",
    ):
        result = assess_statement(statement)

        assert result == {"status": "no_confirmed_issue", "issues": []}


def test_large_or_grouped_bounds_are_not_truncated_into_false_contradictions():
    for statement in (
        "Constraints\n3 <= n <= 2*10^5",
        "Constraints\n3 <= n <= 1,000",
        "Constraints\n3 <= n <= 1 000",
        "Constraints\n3 <= n <= 1 {,} 000",
        "Constraints\n3 <= n <= 2\\cdot 10^5",
    ):
        result = assess_statement(statement)

        assert result == {"status": "no_confirmed_issue", "issues": []}


def test_unresolved_image_placeholder_is_quarantined():
    statement = "Use the formula <image> and print the result."

    result = assess_statement(statement)

    assert result["status"] == "quarantine"
    assert result["issues"][0] == {
        "reason": "unresolved_image_placeholder",
        "evidence": "<image>",
        "start": statement.index("<image>"),
        "end": statement.index("<image>") + len("<image>"),
    }


def test_image_reference_without_an_explicit_placeholder_needs_review():
    statement = "Read the value from the diagram above and print it."

    result = assess_statement(statement)

    assert result["status"] == "review_needed"
    assert result["issues"][0]["reason"] == "external_visual_reference"
    assert result["issues"][0]["evidence"] == "diagram above"


def test_malformed_math_delimiter_needs_review_with_exact_evidence():
    statement = "Compute the value $x + 1."

    result = assess_statement(statement)

    assert result["status"] == "review_needed"
    assert result["issues"] == [
        {
            "reason": "unbalanced_math_delimiter",
            "evidence": "$",
            "start": statement.index("$"),
            "end": statement.index("$") + 1,
        }
    ]


def test_output_contracts_incompatible_with_exact_tokens_are_quarantined():
    result = assess_statement("Print any valid sequence in any order.")

    assert result["status"] == "quarantine"
    assert result["issues"][0]["reason"] == "non_unique_output"
    assert result["issues"][0]["evidence"] == "Print any"


def test_tolerance_and_case_insensitive_contracts_are_quarantined():
    for statement, reason in (
        ("The absolute error must be at most 1e-6.", "numeric_tolerance"),
        ("The judge is case-insensitive.", "case_insensitive_checker_required"),
    ):
        result = assess_statement(statement)

        assert result["status"] == "quarantine"
        assert result["issues"][0]["reason"] == reason


def test_any_value_in_input_does_not_create_an_output_issue():
    result = assess_statement("The input can contain any integers. Print their sum.")

    assert result == {"status": "no_confirmed_issue", "issues": []}


def test_empty_statement_is_quarantined_as_missing_source_text():
    result = assess_statement(" \n ")

    assert result["status"] == "quarantine"
    assert result["issues"] == [
        {"reason": "empty_statement", "evidence": "", "start": 0, "end": 0}
    ]


def test_unknown_or_hard_statement_is_not_blacklisted_by_default():
    statement = "Given a finite automaton A, determine whether it accepts w."

    result = assess_statement(statement)

    assert result == {"status": "no_confirmed_issue", "issues": []}
