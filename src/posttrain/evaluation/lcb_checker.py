"""Task-specific checkers for the frozen LiveCodeBench stdio subset."""
from __future__ import annotations


_MAX_N = 200_000
_MAX_M = 200_000
_MAX_STDIN_CHARS = 8 * 1024 * 1024
_MAX_STDOUT_CHARS = 1024 * 1024
_ASCII_WHITESPACE = " \t\r\n\v\f"


class _ParityDSU:
    """Disjoint set with xor parity from each node to its component root."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size
        self.xor_to_parent = [0] * size

    def _find(self, node: int) -> tuple[int, int]:
        parent = self.parent[node]
        if parent == node:
            return node, 0
        root, parent_parity = self._find(parent)
        self.xor_to_parent[node] ^= parent_parity
        self.parent[node] = root
        return root, self.xor_to_parent[node]

    def constrain(self, left: int, right: int, parity: int) -> bool:
        """Add value[left] xor value[right] == parity; return consistency."""
        left_root, left_parity = self._find(left)
        right_root, right_parity = self._find(right)
        if left_root == right_root:
            return left_parity ^ right_parity == parity

        root_parity = left_parity ^ right_parity ^ parity
        if self.size[left_root] < self.size[right_root]:
            self.parent[left_root] = right_root
            self.xor_to_parent[left_root] = root_parity
            self.size[right_root] += self.size[left_root]
        else:
            self.parent[right_root] = left_root
            self.xor_to_parent[right_root] = root_parity
            self.size[left_root] += self.size[right_root]
        return True


def _read_uint(token: str, field: str, maximum: int) -> int:
    if not token or not token.isascii() or not token.isdigit():
        raise ValueError(f"invalid_arc188_input_{field}")
    normalized = token.lstrip("0") or "0"
    if len(normalized) > len(str(maximum)):
        raise ValueError(f"arc188_input_{field}_out_of_range")
    value = int(normalized)
    if value > maximum:
        raise ValueError(f"arc188_input_{field}_out_of_range")
    return value


def _parse_arc188_input(stdin: str) -> tuple[int, list[tuple[int, int, int]]]:
    if not isinstance(stdin, str):
        raise ValueError("arc188_stdin_must_be_text")
    if len(stdin) > _MAX_STDIN_CHARS:
        raise ValueError("arc188_stdin_too_large")
    if not stdin.isascii():
        raise ValueError("arc188_stdin_must_be_ascii")

    tokens = stdin.split()
    if len(tokens) < 2:
        raise ValueError("arc188_input_missing_header")
    n = _read_uint(tokens[0], "n", _MAX_N)
    m = _read_uint(tokens[1], "m", _MAX_M)
    if n < 2 or m > min(_MAX_M, n * (n - 1)):
        raise ValueError("arc188_input_header_out_of_range")
    if len(tokens) != 2 + 3 * m:
        raise ValueError("arc188_input_token_count_mismatch")

    edges: list[tuple[int, int, int]] = []
    seen_pairs: set[int] = set()
    offset = 2
    for _ in range(m):
        a = _read_uint(tokens[offset], "a", n)
        b = _read_uint(tokens[offset + 1], "b", n)
        c = _read_uint(tokens[offset + 2], "c", 1)
        offset += 3
        if a == 0 or b == 0 or a == b:
            raise ValueError("arc188_input_vertex_out_of_range")
        pair = (a - 1) * n + (b - 1)
        if pair in seen_pairs:
            raise ValueError("arc188_input_duplicate_testimony_pair")
        seen_pairs.add(pair)
        edges.append((a - 1, b - 1, c))
    return n, edges


def check_arc188_c(stdin: str, candidate_stdout: str) -> bool:
    """Check an ARC188 C answer without running reference or candidate code.

    Damaged or out-of-contract input raises ``ValueError``.  Malformed or
    mathematically invalid candidate output returns ``False``.  The checker
    uses at most 2*N parity variables and M testimony edges, bounded by the
    original task constraints.
    """
    n, edges = _parse_arc188_input(stdin)

    if not isinstance(candidate_stdout, str):
        return False
    if len(candidate_stdout) > _MAX_STDOUT_CHARS or not candidate_stdout.isascii():
        return False
    answer = candidate_stdout.strip(_ASCII_WHITESPACE)
    if answer == "-1":
        wants_no_solution = True
    elif len(answer) == n and all(character in "01" for character in answer):
        wants_no_solution = False
    else:
        return False

    # s[i] is whether villager i's testimony is true; h[i] is whether i is
    # honest.  Each testimony imposes s[A] xor h[B] == C.
    dsu = _ParityDSU(2 * n)
    for speaker, subject, claimed_liar in edges:
        if not dsu.constrain(speaker, n + subject, claimed_liar):
            return wants_no_solution

    if wants_no_solution:
        # If the testimony system is consistent, choosing c[i] = s[i] xor h[i]
        # always gives a confused set, so -1 would be incorrect.
        return False

    # A valid candidate fixes c[i] = s[i] xor h[i] for every villager.
    return all(
        dsu.constrain(i, n + i, ord(bit) - ord("0"))
        for i, bit in enumerate(answer)
    )
