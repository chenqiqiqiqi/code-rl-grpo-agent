import json
from pathlib import Path

from posttrain.common import digest
from posttrain.data.constraint_engine import validate_input
from posttrain.data.formal_input_contracts import compile_formal_contract
from posttrain.data.reviewed_contract_selection import select_reviewed_contract


def test_raw_scalar_row_uses_explicit_integer_type_and_constraint_evidence():
    statement = """Constraints
1 <= N <= 5
1 <= M <= 4
All values in input are integers.

Input
Input is given from Standard Input in the following format:
N M

Output
Print N + M.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "complete"
    assert contract["compiler"] == "formal_v1"
    assert contract["sound_for_deletion"] is True
    assert validate_input(contract, "2 4\n")["violations"] == []

    result = validate_input(contract, "0 4\n")
    assert [item["reason"] for item in result["violations"]] == ["integer_out_of_range"]
    assert any("1 <= N <= 5" in ev["text"] for ev in result["violations"][0]["evidence"])


def test_indexed_integer_array_uses_explicit_terminal_index_as_cardinality():
    statement = """Constraints
1 <= N <= 3
1 <= A_i <= 9
All values in input are integers.

Input
Input is given from Standard Input in the following format:
N
A_1 A_2 ... A_N

Output
Print the array.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "complete"
    assert validate_input(contract, "3\n1 8 9\n")["violations"] == []

    result = validate_input(contract, "2\n1 10\n")
    assert [item["reason"] for item in result["violations"]] == ["integer_out_of_range"]
    assert result["violations"][0]["token_index"] == 2


def test_numeric_bounds_without_an_integer_type_guarantee_do_not_reject_decimals():
    statement = """Constraints
1 <= N <= 5

Input
Input is given from Standard Input in the following format:
N

Output
Print N.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    result = validate_input(contract, "1.5\n")
    assert result["violations"] == []


def test_local_integer_declarations_can_type_only_the_named_scalar_and_sequence():
    statement = """N is an integer. A is an integer sequence of length N.

Constraints
1 <= N <= 3
1 <= A_i <= 9

Input
Input is given from Standard Input in the following format:
N
A_1 A_2 ... A_N

Output
Print the sequence.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "complete"
    assert validate_input(contract, "2\n1 9")["violations"] == []
    result = validate_input(contract, "2\n1 10")
    assert result["violations"][0]["reason"] == "integer_out_of_range"
    evidence = result["violations"][0]["evidence"]
    assert any("integer sequence" in item["text"] for item in evidence)


def test_constraints_without_a_format_table_remain_unsupported():
    statement = """Constraints
1 <= N <= 5
All values in input are integers.

Input
Read values and print an answer.

Output
Print the answer.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "unsupported"
    assert contract["sound_for_deletion"] is False
    assert validate_input(contract, "99")["violations"] == []


def test_unhandled_body_guarantee_keeps_a_recognized_table_partial():
    statement = """N is guaranteed to be even.

Constraints
2 <= N <= 10
All values in input are integers.

Input
Input is given from Standard Input in the following format:
N

Output
Print N.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert validate_input(contract, "3")["violations"] == []


def test_unsupported_table_row_stops_later_field_binding():
    statement = """Constraints
1 <= N <= 3
1 <= M <= 3
All values in input are integers.

Input
Input is given from Standard Input in the following format:
N
special encoding for the next field
M

Output
Print M.
"""

    contract = compile_formal_contract(statement)
    result = validate_input(contract, "1\nnot-an-integer\n99\n")
    assert result["coverage"] == "partial"
    assert not any(item["reason"] == "integer_out_of_range" for item in result["violations"])


def test_uncompiled_equality_constraint_never_claims_deletion_trust():
    statement = """All values in input are integers.

Constraints
1 <= A <= 9
1 <= B <= 9
A + B = 10

Input
Input is given from Standard Input in the following format:
A B

Output
Print A + B.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is False
    assert "uncompiled_relation_or_formula" in contract["limitations"]


def test_conditional_or_indexed_relations_are_not_trusted_as_complete():
    statement = """All values in input are integers.
If N is 1, A_1 must be 0.

Constraints
1 <= N <= 3
0 <= A_i <= 9

Input
Input is given from Standard Input in the following format:
N
A_1 A_2 ... A_N

Output
Print the values.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is False
    assert validate_input(contract, "1\n1")["violations"] == []


def test_indexed_bound_depending_on_another_array_element_is_not_complete():
    statement = """All values in input are integers.

Constraints
1 <= N <= 3
0 <= A_i <= 9
A_i <= A_j

Input
Input is given from Standard Input in the following format:
N
A_1 A_2 ... A_N

Output
Print the values.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is False
    assert contract["limitations"]


def test_bullet_prefixed_lower_bound_is_diagnostic_until_its_marker_is_disambiguated():
    statement = """All values in input are integers.

Constraints
- 1 <= N <= 5

Input
Input is given from Standard Input in the following format:
N

Output
Print N.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is False
    # Current parsing can read the list marker as a minus sign. The release
    # gate must not let that uncertainty authorize removal.
    assert "possible_list_marker_bound_ambiguity" in contract["limitations"]


def test_one_sided_field_bound_does_not_claim_complete_input_coverage():
    statement = """All values in input are integers.

Constraints
N <= 10

Input
Input is given from Standard Input in the following format:
N

Output
Print N.
"""

    contract = compile_formal_contract(statement)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is False
    assert "some_input_fields_lack_a_compiled_lower_or_upper_bound" in contract["limitations"]


def test_reviewed_partial_contract_is_hash_bound_and_keeps_only_proven_range_checks():
    statement = r'''Two children are playing tag on a number line. (In the game of tag, the child called "it" tries to catch the other child.) The child who is "it" is now at coordinate A, and he can travel the distance of V per second. The other child is now at coordinate B, and she can travel the distance of W per second.

He can catch her when his coordinate is the same as hers. Determine whether he can catch her within T seconds (including exactly T seconds later). We assume that both children move optimally.

Constraints

* -10^9 \leq A,B \leq 10^9
* 1 \leq V,W \leq 10^9
* 1 \leq T \leq 10^9
* A \neq B
* All values in input are integers.

Input

Input is given from Standard Input in the following format:


A V
B W
T


Output

If "it" can catch the other child, print `YES`; otherwise, print `NO`.

Examples

Input

1 2
3 1
3


Output

YES


Input

1 2
3 2
3


Output

NO


Input

1 2
3 3
3


Output

NO'''
    manifest_path = Path(__file__).parents[1] / "manifests" / "reviewed_formal_contracts_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reviewed = manifest["contracts_by_statement_sha256"]
    entry = reviewed[digest(statement)]
    assert entry["problem_id"] == "taco-train-005296"

    contract = select_reviewed_contract(statement, reviewed)
    assert contract["coverage"] == "partial"
    assert contract["sound_for_deletion"] is True
    assert validate_input(contract, "-1000000000 1 1000000000 1 1000000000")['violations'] == []

    result = validate_input(contract, "1000000001 1 2 1 1")
    assert [item["reason"] for item in result["violations"]] == ["integer_out_of_range"]
    evidence = result["violations"][0]["evidence"]
    assert any("10^9" in item["text"] for item in evidence)


def test_reviewed_contract_keeps_the_entire_upper_bound_after_plus_nine():
    statement = r'''Given positive integers N, K and M, solve the following problem for every integer x between 1 and N (inclusive):
 - Find the number, modulo M, of non-empty multisets containing between 0 and K (inclusive) instances of each of the integers 1, 2, 3 \cdots, N such that the average of the elements is x.

-----Constraints-----
 - 1 \leq N, K \leq 100
 - 10^8 \leq M \leq 10^9 + 9
 - M is prime.
 - All values in input are integers.

-----Input-----
Input is given from Standard Input in the following format:
N K M

-----Output-----
Use the following format:
c_1
c_2
:
c_N

Here, c_x should be the number, modulo M, of multisets such that the average of the elements is x.

-----Sample Input-----
3 1 998244353

-----Sample Output-----
1
3
1

Consider non-empty multisets containing between 0 and 1 instance(s) of each of the integers between 1 and 3. Among them, there are:
 - one multiset such that the average of the elements is k = 1: \{1\};
 - three multisets such that the average of the elements is k = 2: \{2\}, \{1, 3\}, \{1, 2, 3\};
 - one multiset such that the average of the elements is k = 3: \{3\}.'''
    manifest_path = Path(__file__).parents[1] / "manifests" / "reviewed_formal_contracts_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reviewed = manifest["contracts_by_statement_sha256"]
    contract = select_reviewed_contract(statement, reviewed)
    assert contract["coverage"] == "partial"
    assert validate_input(contract, "1 1 1000000007")['violations'] == []
    assert validate_input(contract, "1 1 1000000009")['violations'] == []
    result = validate_input(contract, "1 1 1000000010")
    assert [item["reason"] for item in result["violations"]] == ["integer_out_of_range"]
    assert any("10^9 + 9" in item["text"] for item in result["violations"][0]["evidence"])
