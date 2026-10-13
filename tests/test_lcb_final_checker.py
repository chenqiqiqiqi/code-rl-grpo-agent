from itertools import product

import pytest

from posttrain.data.schema import TestCase as StdioCase
from posttrain.evaluation.lcb_checker import check_arc188_c
from posttrain.evaluation.lcb_results import adjudicate_lcb
from posttrain.execution.contracts import CaseResult, Status


ARC188_SAMPLE_1 = """3 3
1 2 1
1 3 0
2 3 0
"""

ARC188_SAMPLE_2 = """3 6
1 2 1
1 3 0
2 1 1
2 3 0
3 1 1
3 2 0
"""


def test_official_sample_and_another_officially_valid_construction_are_accepted():
    assert check_arc188_c(ARC188_SAMPLE_1, "010\n")
    # AtCoder's statement permits any consistent confused set.  Here 1 and 2
    # are confused liars and 3 is an unconfused honest villager.
    assert check_arc188_c(ARC188_SAMPLE_1, " 110 \t\n")
    assert not check_arc188_c(ARC188_SAMPLE_1, "000\n")
    assert not check_arc188_c(ARC188_SAMPLE_1, "-1\n")


def test_minus_one_is_accepted_only_when_no_assignment_exists():
    assert check_arc188_c(ARC188_SAMPLE_2, "-1\n")
    assert not check_arc188_c(ARC188_SAMPLE_2, "000\n")
    assert check_arc188_c("3 0\n", "000\n")
    assert not check_arc188_c("3 0\n", "-1\n")


def test_candidate_must_be_exactly_one_ascii_output_token():
    stdin = "3 0\n"
    for output in (
        "",
        "010 110\n",
        "010\n010\n",
        "01x\n",
        "0010\n",
        "-1 extra\n",
        "\u00a0010\u00a0",
    ):
        assert not check_arc188_c(stdin, output), repr(output)


def test_corrupt_or_out_of_contract_stdin_raises_value_error():
    for stdin in (
        "",
        "1 0\n",
        "3 -1\n",
        "3 0\n1\n",
        "3 1\n1 1 0\n",
        "3 1\n1 2 2\n",
        "3 2\n1 2 0\n1 2 1\n",
        "3\u00a00\n",
    ):
        with pytest.raises(ValueError):
            check_arc188_c(stdin, "000\n")


def test_oversized_input_is_rejected_and_oversized_candidate_fails_closed():
    with pytest.raises(ValueError, match="too_large"):
        check_arc188_c(" " * (8 * 1024 * 1024 + 1), "000\n")
    assert not check_arc188_c("3 0\n", "0" * (1024 * 1024 + 1))


def _serialize_instance(n: int, edges: list[tuple[int, int, int]]) -> str:
    lines = [f"{n} {len(edges)}"]
    lines.extend(f"{a + 1} {b + 1} {c}" for a, b, c in edges)
    return "\n".join(lines) + "\n"


def _brute_feasible(n: int, edges: list[tuple[int, int, int]], confused: int) -> bool:
    # Independent definition-level oracle: enumerate honesty assignments and
    # compare each speaker's actual truthfulness with the claimed status.
    for honesty in product((0, 1), repeat=n):
        if all(
            (honesty[a] ^ ((confused >> a) & 1)) == (honesty[b] ^ c)
            for a, b, c in edges
        ):
            return True
    return False


def test_matches_exhaustive_bruteforce_for_every_three_villager_statement_graph():
    n = 3
    directed_pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    graph_states = (None, 0, 1)

    for state in product(graph_states, repeat=len(directed_pairs)):
        edges = [
            (a, b, c)
            for (a, b), c in zip(directed_pairs, state)
            if c is not None
        ]
        stdin = _serialize_instance(n, edges)
        feasible_masks = {
            confused
            for confused in range(1 << n)
            if _brute_feasible(n, edges, confused)
        }

        for confused in range(1 << n):
            candidate = "".join(str((confused >> i) & 1) for i in range(n))
            assert check_arc188_c(stdin, candidate) == (confused in feasible_masks), (
                edges,
                candidate,
            )
        assert check_arc188_c(stdin, "-1") == (not feasible_masks), edges


def test_final_adjudicator_uses_checker_and_keeps_every_raw_case():
    tests = (
        StdioCase(ARC188_SAMPLE_1, "010\n"),
        StdioCase(ARC188_SAMPLE_1, "010\n"),
    )
    results = (
        CaseResult(Status.OK, stdout="110\n"),
        CaseResult(Status.OK, stdout="010\n"),
    )

    row = adjudicate_lcb("atcoder:arc188_c", tests, results)

    assert row["raw_test_count"] == 2
    assert row["passed"] == 2
    assert row["pass_at_1"] is True
    assert [case["original_test_index"] for case in row["cases"]] == [0, 1]


def test_final_adjudicator_propagates_damaged_input_as_data_error():
    with pytest.raises(ValueError):
        adjudicate_lcb(
            "atcoder:arc188_c",
            (StdioCase("not a valid ARC188 instance", "010\n"),),
            (CaseResult(Status.OK, stdout="010\n"),),
        )
