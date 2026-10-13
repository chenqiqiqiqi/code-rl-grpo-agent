"""Diagnostic statement-derived contract proposals for stdio input domains.

The compiler only emits rules when their input binding is explicit.  It does
not execute problem code and it never evaluates source expressions with eval.
Unknown syntax is intended to lower coverage, but review found incomplete prose
grammar recognition. Compiler proposals (including ``complete``) are NOT release
approval. Production cleaning uses exact-statement frozen reviewed contracts via
``reviewed_contract_selection``. Never delete tests from generic proposals alone.
"""

from __future__ import annotations

import re
from math import gcd
from collections import defaultdict
from posttrain.common import digest


SCHEMA_VERSION = 1

_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,2}[ \t]*)?"
    r"(?:[-*_]{2,}[ \t]*)?"
    r"(?P<name>input(?:[ \t]+format)?|constraints?|output|examples?|"
    r"sample[ \t]+input|sample[ \t]+output|notes?|explanation|subtasks?)"
    r"(?:[ \t]*[-*_]{2,})?(?:[*_]{1,2})?[ \t]*:?[ \t]*$"
)
_BOUNDARY_HEADERS = {"output", "examples", "example", "sample input", "sample output",
                     "note", "notes", "explanation", "subtasks", "constraints", "constraint"}

_NAME = r"(?:\|[A-Za-z][A-Za-z0-9]*\||[A-Za-z][A-Za-z0-9]*(?:_\{?[A-Za-z0-9]+\}?)?)"
_SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_POW_EXP = r"(?:\^\s*\{?\d+\}?|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
_GROUPED_INTEGER = r"(?:\d{1,3}(?:(?:[ \t,]|\\,)\d{3})+)"
_NUM_TOKEN = (
    r"[+-]?\s*(?:" + _GROUPED_INTEGER + r"|\d+)\s*(?:"
    r"(?:(?:\\times|\\cdot|×|·|⋅|\*)\s*10\s*" + _POW_EXP + r")|"
    r"(?:\^\s*\{?\d+\}?)"
    r")?(?![A-Za-z0-9_^*×·⋅\\])(?![.,]\d)"
)
_ENDPOINT = r"(?:" + _NUM_TOKEN + r"|" + _NAME + r"(?:\s*[+-]\s*(?:" + _NAME + r"|" + _NUM_TOKEN + r"))*)"
_VAR_LIST = _NAME + r"(?:\s*(?:,|\band\b)\s*" + _NAME + r")*"
_CMP = r"(?:\\leq?|\\geq?|<=|>=|≤|≥|<|>)"
_RANGE_RE = re.compile(
    r"(?P<left>" + _ENDPOINT + r")\s*(?P<op1>" + _CMP + r")\s*"
    r"(?P<vars>" + _VAR_LIST + r")\s*(?P<op2>" + _CMP + r")\s*"
    r"(?P<right>" + _ENDPOINT + r")"
)
_LOWER_RE = re.compile(r"(?P<left>" + _NUM_TOKEN + r")\s*(?P<op><=|≤|<|\\leq?)\s*(?P<var>" + _NAME + r")")
_UPPER_RE = re.compile(r"(?P<var>" + _NAME + r")\s*(?P<op><=|≤|<|\\leq?)\s*(?P<right>" + _NUM_TOKEN + r")")
_CROSS_RE = re.compile(
    r"(?P<expr>" + _NAME + r"(?:\s*[+-]\s*(?:" + _NAME + r"|\d+))+?)"
    r"\s*(?P<op><=|≤|<|\\leq?)\s*(?P<right>" + _NUM_TOKEN + r")"
)
_INEQUALITY_RE = re.compile(r"(?:\\leq?|\\geq?|<=|>=|≤|≥|<|>|≠|!=)")

_LINE_MARKER_RE = re.compile(
    r"\b(?:the\s+)?(?:(?:first|second|third|fourth|fifth|sixth|last)\s+(?:input\s+)?line(?:s)?|"
    r"(?:(?:each|every)\s+of\s+the\s+)?(?:next|following)\s+[^\n.;,]{1,40}\s+lines?|"
    r"(?:each|every)\s+of\s+the\s+next\s+[^\n.;,]{1,40}\s+lines?)\b",
    re.IGNORECASE,
)
_VARIABLE_TOKEN_RE = re.compile(_NAME)
_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}
_CONJUNCTIONS = {"and", "or", "to", "of", "where", "which", "that", "with"}


def _header_name(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def _sections(statement: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    headers = []
    for match in _HEADER_RE.finditer(statement):
        headers.append((match.start(), match.end(), _header_name(match.group("name"))))

    input_header = next((item for item in headers if item[2] in {"input", "input format"}), None)
    if input_header is None:
        return None, None

    input_end = len(statement)
    for start, _end, name in headers:
        if start > input_header[0] and name in _BOUNDARY_HEADERS:
            input_end = start
            break
    input_span = (input_header[1], input_end)

    constraints = [item for item in headers if item[2] in {"constraint", "constraints"}]
    constraint_span = None
    if constraints:
        before = [item for item in constraints if item[0] < input_header[0]]
        after = [item for item in constraints if input_header[1] <= item[0] <= input_end]
        header = before[-1] if before else (after[0] if after else None)
        if header is not None:
            end = input_header[0] if header[0] < input_header[0] else len(statement)
            for start, _end, name in headers:
                if header[0] < start < end and name in _BOUNDARY_HEADERS - {"constraints", "constraint"}:
                    end = start
                    break
            constraint_span = (header[1], end)
    return input_span, constraint_span


def _evidence(statement: str, start: int, end: int, role: str) -> dict:
    return {"start": start, "end": end, "text": statement[start:end], "role": role}


def _normalize_name(name: str, *, indexed: bool = False) -> str:
    result = name.strip().strip("$").replace("\\", "").replace("{", "").replace("}", "")
    if result.startswith("|") and result.endswith("|"):
        result = result[1:-1]
    if indexed:
        result = re.sub(r"_(?:\d+|i)", "_i", result)
    return result


def _parse_number(raw: str) -> int | None:
    original = raw.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    original = re.sub(r"\\(?:,|;|!|quad|qquad)", "", original)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", original):
        return int(original.replace(",", ""))
    if re.fullmatch(r"[+-]?\d{1,3}(?:[ \t]+\d{3})+", original):
        return int(re.sub(r"[ \t]", "", original))
    text = original
    text = re.sub(r"\s+", "", text)
    sign = -1 if text.startswith("-") else 1
    unsigned = text.lstrip("+-")

    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    superscript = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", unsigned)
    if superscript:
        base = int(superscript.group(1))
        exponent = int(superscript.group(2).translate(superscript_map))
        if exponent > 1000:
            return None
        return sign * (base ** exponent)

    power = re.fullmatch(r"(\d+)\^\{?(\d+)\}?", unsigned)
    if power:
        exponent = int(power.group(2))
        if exponent > 1000:
            return None
        return sign * (int(power.group(1)) ** exponent)

    product = re.fullmatch(r"(\d+)(?:\\times|\\cdot|×|·|⋅|\*)(10)(?:\^\{?(\d+)\}?|([⁰¹²³⁴⁵⁶⁷⁸⁹]+))", unsigned)
    if product:
        exponent = int(product.group(3) or product.group(4).translate(superscript_map))
        if exponent > 1000:
            return None
        return sign * int(product.group(1)) * (10 ** exponent)

    if re.fullmatch(r"\d+", unsigned):
        # A flattened 10^k is indistinguishable from a literal such as 109.
        # Leave the bound uncompiled instead of silently choosing a meaning.
        if re.fullmatch(r"10[3-9]", unsigned):
            return None
        return sign * int(unsigned)
    return None


def _parse_linear(raw: str) -> dict | None:
    """Parse a deliberately small sum/difference grammar; never eval source."""
    text = raw.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:,|;|!|quad|qquad)", "", text)
    text = text.replace("−", "-").strip()
    if not text:
        return None
    whole_number = _parse_number(text)
    if whole_number is not None:
        return {"constant": whole_number, "terms": {}}

    atom = r"(?:10\s*\^\s*\{?\d+\}?|\d+|" + _NAME + r")"
    atom_re = re.compile(atom)
    pos = 0
    terms: dict[str, int] = defaultdict(int)
    constant = 0
    count = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        sign = 1
        if text[pos] in "+-":
            sign = -1 if text[pos] == "-" else 1
            pos += 1
            while pos < len(text) and text[pos].isspace():
                pos += 1
        elif count:
            # Adjacent atoms without an arithmetic operator are not a sum.
            return None
        match = atom_re.match(text, pos)
        if not match:
            return None
        token = match.group(0).strip()
        number = _parse_number(token)
        if number is not None:
            constant += sign * number
        elif re.fullmatch(_NAME, token):
            terms[_normalize_name(token)] += sign
        else:
            return None
        pos = match.end()
        count += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] not in "+-":
            return None
    if count == 0:
        return None
    return {"constant": constant, "terms": {key: value for key, value in sorted(terms.items()) if value}}


def _eval_linear(expression: dict, values: dict) -> int | None:
    total = expression.get("constant", 0)
    for name, coefficient in expression.get("terms", {}).items():
        value = values.get(name)
        if not isinstance(value, int):
            return None
        total += coefficient * value
    return total


def _adjust(expression: dict, delta: int) -> dict:
    result = {"constant": expression["constant"] + delta, "terms": dict(expression["terms"])}
    return result


def _canonical_cmp(operator: str) -> str:
    return {"<=": "≤", "\\le": "≤", "\\leq": "≤", ">=": "≥",
            "\\ge": "≥", "\\geq": "≥"}.get(operator, operator)


def _range_rules(statement: str, spans: list[tuple[int, int, str]]) -> tuple[dict, set[tuple[int, int]], bool]:
    rules: dict[str, list[dict]] = defaultdict(list)
    recognized: set[tuple[int, int]] = set()
    unsupported_inequality = False

    for base, end, role in spans:
        text = statement[base:end]
        for match in _RANGE_RE.finditer(text):
            left = _parse_linear(match.group("left"))
            right = _parse_linear(match.group("right"))
            if left is None or right is None:
                continue
            op1 = _canonical_cmp(match.group("op1"))
            op2 = _canonical_cmp(match.group("op2"))
            if (op1 in {"≤", "<"}) != (op2 in {"≤", "<"}):
                # A turning chain has two lower or two upper bounds. Do not
                # overwrite one with the other; leave it unresolved.
                continue
            variables = [_normalize_name(x, indexed=True) for x in _VARIABLE_TOKEN_RE.findall(match.group("vars"))
                         if x.lower() not in _CONJUNCTIONS]
            if not variables:
                continue
            lower = None
            upper = None
            if op1 in {"≤", "<"}:
                lower = _adjust(left, 1) if op1 == "<" else left
            elif op1 in {"≥", ">"}:
                upper = _adjust(left, -1) if op1 == ">" else left
            if op2 in {"≤", "<"}:
                upper = _adjust(right, -1) if op2 == "<" else right
            elif op2 in {"≥", ">"}:
                lower = _adjust(right, 1) if op2 == ">" else right
            if lower is None and upper is None:
                continue
            ev_start, ev_end = base + match.start(), base + match.end()
            ev = _evidence(statement, ev_start, ev_end, role)
            recognized.add((ev_start, ev_end))
            for name in variables:
                rules[name].append({"min_expr": lower, "max_expr": upper, "evidence": [ev]})

        # Simple one-sided integer bounds such as "1 <= n" or "n <= 100".
        for regex, side in ((_LOWER_RE, "lower"), (_UPPER_RE, "upper")):
            for match in regex.finditer(text):
                ev_start, ev_end = base + match.start(), base + match.end()
                if any(start <= ev_start and ev_end <= finish for start, finish in recognized):
                    continue
                raw = match.group("left") if side == "lower" else match.group("right")
                expression = _parse_linear(raw)
                if expression is None:
                    continue
                op = _canonical_cmp(match.group("op"))
                if side == "lower" and op == "<":
                    expression = _adjust(expression, 1)
                if side == "upper" and op == "<":
                    expression = _adjust(expression, -1)
                name = _normalize_name(match.group("var"), indexed=True)
                ev = _evidence(statement, ev_start, ev_end, role)
                recognized.add((ev_start, ev_end))
                rules[name].append({
                    "min_expr": expression if side == "lower" else None,
                    "max_expr": expression if side == "upper" else None,
                    "evidence": [ev],
                })

        # Cross-variable limits are accepted only as sums/differences of names.
        for match in _CROSS_RE.finditer(text):
            expression = _parse_linear(match.group("expr"))
            limit = _parse_number(match.group("right"))
            if expression is None or limit is None or len(expression["terms"]) < 2:
                continue
            op = match.group("op")
            ev_start, ev_end = base + match.start(), base + match.end()
            ev = _evidence(statement, ev_start, ev_end, role)
            recognized.add((ev_start, ev_end))
            rules["__cross__"].append({"expression": expression, "op": op, "limit": limit, "evidence": [ev]})

        for match in _INEQUALITY_RE.finditer(text):
            absolute = (base + match.start(), base + match.end())
            if not any(start <= absolute[0] and absolute[1] <= finish for start, finish in recognized):
                # Non-domain prose may contain comparisons; only count this as
                # unresolved in the constraints block or inside an input field.
                unsupported_inequality = True

    return dict(rules), recognized, unsupported_inequality


def _expr_from_count(raw: str) -> dict | None:
    return _parse_linear(raw)


def _field_names(raw: str, count: int) -> list[str]:
    # Only read the declaration list before bounds or explanatory prose.
    cut = len(raw)
    for marker in ("(", "—", "–", ":", ";", "."):
        found = raw.find(marker)
        if found >= 0:
            cut = min(cut, found)
    declaration = raw[:cut]
    found = []
    for token in _VARIABLE_TOKEN_RE.findall(declaration):
        if token.lower() in _CONJUNCTIONS:
            continue
        name = _normalize_name(token)
        if name not in found:
            found.append(name)
        if len(found) >= count:
            break
    return found


def _count_prefix(prefix: str) -> tuple[int | None, dict | None, bool]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*|\d+|[+-]", prefix.strip())
    if not words:
        return None, None, False
    last = words[-1].lower()
    if last in _COUNT_WORDS:
        return _COUNT_WORDS[last], None, False
    # A variable or simple arithmetic expression before "integers" is an
    # array cardinality only when a separate element variable follows it.
    candidate = re.sub(r"^(?:the\s+|a\s+|an\s+)", "", prefix.strip(), flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:space[- ]separated\s+)", "", candidate, flags=re.IGNORECASE)
    expression = _expr_from_count(candidate)
    if expression is not None:
        return None, expression, True
    return None, None, False


def _line_segments(statement: str, start: int, end: int) -> list[tuple[int, int, str]]:
    text = statement[start:end]
    markers = list(_LINE_MARKER_RE.finditer(text))
    segments = []
    for index, match in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        segment_start = start + match.start()
        segment_end = start + stop
        segments.append((segment_start, segment_end, statement[segment_start:segment_end]))
    return segments


def _parse_segment(segment: str, absolute_start: int, statement: str, range_map: dict, index: int) -> tuple[dict | None, bool]:
    contains = re.search(r"\bcontains?\b", segment, re.IGNORECASE)
    if not contains:
        return None, True
    repeated = re.match(r"(?is)^\s*(?:each\s+of\s+the\s+next|the\s+next|the\s+following)\s+(.+?)\s+lines?\b", segment)
    repeat_count = _expr_from_count(repeated.group(1)) if repeated else None

    int_match = re.search(r"\bintegers?\b|\binteger\b", segment[contains.end():], re.IGNORECASE)
    if int_match:
        int_start = contains.end() + int_match.start()
        int_end = contains.end() + int_match.end()
        prefix = segment[contains.end():int_start]
        # Do not jump across a clause/sentence to a later declaration. For
        # example, the leading value in "number of test cases. Followed by
        # two integers A and B" is T, not A. Unknown layout stops the prefix.
        if re.search(r'[.;:,]|\b(?:test\s*cases?|followed|then|each|datasets?)\b',prefix,re.I):
            return None, True
        after = segment[int_end:]
        if re.search(r'\b(?:string|character|real|float|double)\b', after, re.I):
            return None, True
        fixed_count, vector_count, count_is_vector = _count_prefix(prefix)
        is_plural = segment[int_start:int_end].lower().endswith("s")
        scalar_count = fixed_count if fixed_count is not None else (1 if not is_plural else None)
        # Repeated row declarations describe a fixed set of fields per row.
        if repeat_count is not None:
            count_expr = repeat_count
            is_vector = False
            field_count = scalar_count
        elif count_is_vector and is_plural:
            count_expr = vector_count
            is_vector = True
            field_count = 1
        elif scalar_count is not None:
            count_expr = None
            is_vector = False
            field_count = scalar_count
        else:
            return None, True

        names = _field_names(after, field_count or 1)
        if is_vector:
            item_name = _normalize_name(names[0], indexed=True) if names else f"item_{index}_i"
            fields = [{"kind": "int", "name": item_name, "bounds": range_map.get(item_name, []),
                       "evidence": []}]
        else:
            count = field_count or 1
            fields = []
            for field_index in range(count):
                name = names[field_index] if field_index < len(names) else f"field_{index}_{field_index}"
                fields.append({"kind": "int", "name": name, "bounds": range_map.get(name, []), "evidence": []})
        declaration_ev = _evidence(statement, absolute_start + contains.start(), absolute_start + int_end, "input_format")
        for field in fields:
            field["evidence"] = [declaration_ev]
            for bound in field["bounds"]:
                field["evidence"].extend(bound["evidence"])
        operation = {"id": f"op{index}", "kind": "repeat" if (is_vector or repeat_count is not None) else "fields",
                     "fields": fields, "evidence": [declaration_ev]}
        if is_vector:
            operation["count_expr"] = count_expr
            operation["array"] = True
            operation["distinct"] = bool(re.search(r"\bdistinct\b|\bdifferent\b", segment, re.IGNORECASE))
        elif repeat_count is not None:
            operation["count_expr"] = repeat_count
            operation["rows"] = True
        return operation, False

    string_match = re.search(r"\bstring\b", segment[contains.end():], re.IGNORECASE)
    if string_match:
        string_start = contains.end() + string_match.start()
        string_end = contains.end() + string_match.end()
        prefix=segment[contains.end():string_start]
        if re.search(r'[.;:,]|\b(?:test\s*cases?|followed|then|each|datasets?)\b',prefix,re.I):
            return None, True
        after = segment[string_end:]
        length_match = re.search(r"\b(?:of\s+)?(?:exactly\s+)?length\s+(" + _NAME + r")\b", after, re.IGNORECASE)
        declaration = after[:length_match.start()] if length_match else after
        names = _field_names(declaration, 1)
        name = names[0] if names else f"string_{index}"
        # An arithmetic suffix must not be silently dropped from a length.
        if length_match and re.match(r'\s*[+*/^−-]', after[length_match.end():]):
            return None, True
        if repeated:
            return None, True
        length_expr = _expr_from_count(length_match.group(1)) if length_match else None
        declaration_ev = _evidence(statement, absolute_start + contains.start(), absolute_start + string_end, "input_format")
        field = {"kind": "string", "name": name, "bounds": range_map.get("|" + name + "|", []) + range_map.get(name, []),
                 "length_expr": length_expr, "evidence": [declaration_ev]}
        for bound in field["bounds"]:
            field["evidence"].extend(bound["evidence"])
        return {"id": f"op{index}", "kind": "fields", "fields": [field], "evidence": [declaration_ev]}, False

    return None, True


def _semantic_checks(statement: str, input_span: tuple[int, int], constraint_span: tuple[int, int] | None,
                     operations: list[dict]) -> tuple[list[dict], bool]:
    spans = [input_span]
    if constraint_span:
        spans.append(constraint_span)
    scoped = "\n".join(statement[a:b] for a, b in spans)
    # Guarantees are often stated in the problem body before Input/Constraints.
    # Treat them as unresolved unless the same exact relation is compiled.
    body = statement[:input_span[0]]
    semantics = []
    unresolved = False

    # Bind permutation only when the input explicitly names the array that is
    # required to be a permutation. A nearby word alone is not enough.
    permutation_bound = False
    vectors = [operation for operation in operations if operation.get("array") and len(operation["fields"]) == 1]
    for op in vectors:
        field_name = op["fields"][0]["name"]
        base_name = re.sub(r"_i$", "", field_name)
        pattern = re.compile(
            r"\b(?:(?:array|sequence|vector)\s+)?\$?" + re.escape(base_name) +
            r"\$?\s+(?:is|forms?)\s+(?:a\s+)?permutation\b", re.IGNORECASE,
        )
        for span_start, span_end in spans:
            match = pattern.search(statement[span_start:span_end])
            if match:
                semantics.append({"id": "semantic:permutation", "kind": "permutation", "operation_id": op["id"],
                                  "field_name": field_name,
                                  "evidence": [_evidence(statement, span_start + match.start(), span_start + match.end(), "input_domain")]})
                permutation_bound = True
                break
        if permutation_bound:
            break

    # Tree constraints need an explicit tree statement plus a recognized
    # n-1 edge-row operation. The vertex count comes from that row count.
    tree_match = None
    tree_span_start = None
    for span_start, span_end in spans:
        match = re.search(r"(?:given|input|resulting|underlying)?\s*(?:graph|road\s+map|network)\s+(?:is|forms?)\s+(?:a\s+)?tree|\bthe\s+given\s+graph\s+is\s+a\s+tree\b",
                          statement[span_start:span_end], re.IGNORECASE)
        if match:
            tree_match = match
            tree_span_start = span_start
            break
    if tree_match:
        edge_ops = [op for op in operations if op.get("rows") and len(op["fields"]) == 2]
        tree_ops = [op for op in edge_ops if op.get("count_expr", {}).get("constant") == -1 and
                    len(op.get("count_expr", {}).get("terms", {})) == 1 and
                    list(op["count_expr"]["terms"].values()) == [1]]
        if tree_ops:
            op = tree_ops[0]
            node_count_var = next(iter(op["count_expr"]["terms"]))
            name = op["fields"][0]["name"]
            other = op["fields"][1]["name"]
            ev = _evidence(statement, tree_span_start + tree_match.start(), tree_span_start + tree_match.end(), "input_domain")
            semantics.append({"id": "semantic:tree", "kind": "tree", "operation_id": op["id"],
                              "node_count_var": node_count_var,
                              "endpoint_names": [name, other], "evidence": [ev]})
        else:
            unresolved = True

    semantic_markers = re.compile(
        r"\b(?:distinct|different|unique|permutation|tree|connected|even|odd|prime|guaranteed|"
        r"non[- ]decreasing|strictly\s+increasing|strictly\s+decreasing|at\s+least|at\s+most|"
        r"nonzero|positive|negative|exactly\s+once)\b",
        re.IGNORECASE,
    )
    handled_terms = set()
    if any(check["kind"] == "permutation" for check in semantics):
        handled_terms.add("permutation")
    if any(check["kind"] == "tree" for check in semantics):
        handled_terms.add("tree")
    for source in (body, scoped):
        for match in semantic_markers.finditer(source):
            word = match.group(0).lower()
            if word not in handled_terms:
                # "distinct" is handled only when the vector declaration itself
                # explicitly applies it to one array.
                if word == "distinct" and any(op.get("distinct") for op in operations):
                    continue
                unresolved = True
    if re.search(r"\bstring\b", scoped, re.IGNORECASE):
        for operation in operations:
            for field in operation["fields"]:
                if field.get("kind") == "string" and (field.get("length_expr") is None or not field.get("alphabet")):
                    unresolved = True
    return semantics, unresolved


def compile_contract(statement: str) -> dict:
    """Compile a conservative JSON-serializable contract from one statement."""
    if not isinstance(statement, str) or not statement:
        return {"schema_version": SCHEMA_VERSION, "coverage": "unsupported", "operations": [],
                "constraints": [], "semantics": [], "limitations": ["missing_statement"]}

    input_span, constraint_span = _sections(statement)
    if input_span is None:
        return {"schema_version": SCHEMA_VERSION, "coverage": "unsupported", "operations": [],
                "constraints": [], "semantics": [], "limitations": ["input_section_not_found"]}

    rule_spans = [(input_span[0], input_span[1], "input_constraint")]
    if constraint_span:
        rule_spans.append((constraint_span[0], constraint_span[1], "constraint_section"))
    range_map, recognized, unresolved_inequality = _range_rules(statement, rule_spans)
    cross_checks = range_map.pop("__cross__", [])

    operations = []
    unparsed_line = False
    segments = _line_segments(statement, *input_span)
    halted = False
    for index, (start, _end, segment) in enumerate(segments):
        if halted:
            break
        operation, unresolved = _parse_segment(segment, start, statement, range_map, index)
        if operation:
            operations.append(operation)
        if unresolved:
            unparsed_line = True
            halted = True

    first_line_recognized = bool(segments and re.search(r"\bfirst\s+(?:input\s+)?line\b", segments[0][2], re.IGNORECASE))
    if segments and not first_line_recognized:
        unparsed_line = True
    input_text = statement[input_span[0]:input_span[1]]
    parsed_contains = sum(bool(re.search(r"\bcontains?\b", segment, re.IGNORECASE)) for _start, _end, segment in segments[:len(operations)])
    if len(re.findall(r"\bcontains?\b", input_text, re.IGNORECASE)) > parsed_contains:
        unparsed_line = True

    # A prose declaration may give useful scalar bounds but cannot safely bind
    # them to stdin unless an input-line declaration was also recovered.
    attached_names = {field["name"] for operation in operations for field in operation["fields"]}
    unattached_rules = [name for name, items in range_map.items() if items and name not in attached_names and name != "__cross__"]
    semantics, unresolved_semantics = _semantic_checks(statement, input_span, constraint_span, operations)

    limitations = []
    if not operations:
        limitations.append("input_layout_not_recognized")
    if unparsed_line:
        limitations.append("input_field_declaration_not_recognized")
    if unattached_rules:
        limitations.append("some_bounds_could_not_be_bound_to_input_fields")
    if unresolved_inequality:
        limitations.append("some_inequalities_not_compiled")
    if unresolved_semantics:
        limitations.append("some_semantic_constraints_not_compiled")
    if re.search(r"\b(?:test cases?|datasets?|until end of input|end of input|sentinel|multiple datasets?)\b",
                 statement[input_span[0]:input_span[1]], re.IGNORECASE):
        limitations.append("repeated_or_terminated_input_grammar_not_compiled")

    has_input_types = any(op["fields"] for op in operations)
    unbounded_fields = [field for operation in operations for field in operation["fields"]
                        if field.get("kind") == "int" and not field.get("bounds")]
    if unbounded_fields:
        limitations.append("some_input_fields_have_no_compiled_range")
    if segments and not first_line_recognized:
        limitations.append("first_input_field_not_recognized")
    if len(re.findall(r"\bcontains?\b", input_text, re.IGNORECASE)) > parsed_contains:
        limitations.append("some_input_declarations_were_not_compiled")
    has_unbound_ranges = bool(unattached_rules)
    complete = bool(operations) and has_input_types and not unbounded_fields and not any((unparsed_line, unresolved_inequality,
                                                                  unresolved_semantics, has_unbound_ranges,
                                                                  limitations and "repeated_or_terminated_input_grammar_not_compiled" in limitations))
    coverage = "complete" if complete else ("partial" if operations or cross_checks else "unsupported")

    # Any variable rule not attached to an input field remains visible for
    # review, but validate_input never guesses where that value occurs.
    constraints = []
    for operation in operations:
        for field in operation["fields"]:
            for bound_index, bound in enumerate(field.get("bounds", [])):
                constraints.append({"id": f"{operation['id']}:{field['name']}:range{bound_index}",
                                    "kind": "range", "field_name": field["name"],
                                    "min_expr": bound["min_expr"], "max_expr": bound["max_expr"],
                                    "evidence": bound["evidence"]})
    for index, check in enumerate(cross_checks):
        constraints.append({"id": f"constraint:cross{index}", "kind": "upper_bound",
                            "expression": check["expression"], "operator": check["op"],
                            "limit": check["limit"], "evidence": check["evidence"]})

    return {"schema_version": SCHEMA_VERSION, "coverage": coverage, "operations": operations,
            "constraints": constraints, "semantics": semantics,
            "limitations": list(dict.fromkeys(limitations))}


def _integer_value(token: str) -> int | None:
    if not re.fullmatch(r"[+-]?[0-9]+", token):
        return None
    sign = "-" if token.startswith("-") else ""
    digits = token.lstrip("+-").lstrip("0") or "0"
    if len(digits) > 4000:
        return None
    try:
        return int(sign + digits)
    except ValueError:
        return None


def _compare_integer_token(token: str, bound: int) -> int:
    """Compare an ASCII integer token to a bound without converting huge input."""
    negative = token.startswith("-")
    digits = token.lstrip("+-").lstrip("0") or "0"
    token_zero = digits == "0"
    if token_zero:
        negative = False
    if bound == 0:
        bound_negative = False
        bound_digits = "0"
    else:
        bound_negative = bound < 0
        bound_digits = str(abs(bound))
    if negative != bound_negative:
        return -1 if negative else 1
    magnitude = (len(digits) > len(bound_digits)) - (len(digits) < len(bound_digits))
    if magnitude == 0:
        magnitude = (digits > bound_digits) - (digits < bound_digits)
    return -magnitude if negative else magnitude


def _huge_integer_violations(field: dict, token: str, scalar_values: dict, token_index: int) -> list[dict]:
    violations = []
    for index, bound in enumerate(field.get("bounds", [])):
        minimum = _eval_linear(bound["min_expr"], scalar_values) if bound.get("min_expr") is not None else None
        maximum = _eval_linear(bound["max_expr"], scalar_values) if bound.get("max_expr") is not None else None
        below = minimum is not None and _compare_integer_token(token, minimum) < 0
        above = maximum is not None and _compare_integer_token(token, maximum) > 0
        if below or above:
            violations.append({"reason": "integer_out_of_range", "rule_id": f"{field.get('name')}:range{index}",
                               "evidence": bound["evidence"], "token_index": token_index, "value": token,
                               "minimum": minimum, "maximum": maximum})
    return violations


def _check_bounds(field: dict, value: int, scalar_values: dict, token_index: int,
                  token: str) -> tuple[list[dict], list[dict]]:
    violations = []
    pending = []
    for index, bound in enumerate(field.get("bounds", [])):
        minimum = _eval_linear(bound["min_expr"], scalar_values) if bound.get("min_expr") is not None else None
        maximum = _eval_linear(bound["max_expr"], scalar_values) if bound.get("max_expr") is not None else None
        unresolved_min = bound.get("min_expr") is not None and minimum is None
        unresolved_max = bound.get("max_expr") is not None and maximum is None
        if unresolved_min or unresolved_max:
            pending.append({"field": field, "value": value, "token_index": token_index, "token": token,
                            "bound_index": index})
            continue
        if minimum is None and maximum is None:
            continue
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            violations.append({"reason": "integer_out_of_range", "rule_id": f"{field.get('name')}:range{index}",
                               "evidence": bound["evidence"], "token_index": token_index, "value": token,
                               "minimum": minimum, "maximum": maximum})
    return violations, pending


def _union_evidence(items: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for item in items:
        for evidence in item.get("evidence", []):
            key = (evidence.get("start"), evidence.get("end"), evidence.get("text"))
            if key not in seen:
                seen.add(key)
                result.append(evidence)
    return result


def _tree_is_valid(n: int, edges: list[tuple[int, int]]) -> bool:
    if n < 1 or len(edges) != n - 1:
        return False
    parent = list(range(n + 1))
    size = [1] * (n + 1)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in edges:
        if left < 1 or left > n or right < 1 or right > n or left == right:
            return False
        a, b = find(left), find(right)
        if a == b:
            return False
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return True


def _string_field_has_token_safe_domain(field: dict, bound_values: dict) -> bool:
    alphabet=field.get("alphabet")
    if not isinstance(alphabet,str) or not alphabet or any(c.isspace() for c in alphabet):
        return False
    length=field.get("length_expr")
    if length is not None:
        minimum=_eval_linear(length,bound_values)
        if minimum is not None:
            return minimum>=1
    minimum_expr=field.get("min_length_expr")
    if minimum_expr is not None:
        minimum=_eval_linear(minimum_expr,bound_values)
        return minimum is not None and minimum>=1
    return False


def validate_input(contract: dict, stdin: str) -> dict:
    if not isinstance(stdin, str) or not isinstance(contract, dict):
        return {'coverage': 'unsupported', 'violations': []}
    tokens = re.findall(r'\S+', stdin)
    if contract.get('schema_version') == 2:
        result = _validate_multicase(contract, tokens)
    else:
        result = _validate_flat_tokens(contract, tokens)
    result = {key: result[key] for key in ('coverage', 'violations')}
    # Pointwise adjudications are separate from grammar coverage. Matching is
    # deliberately byte-for-byte at the canonical string-hash level: no fuzzy
    # matching, whitespace normalization or propagation to unreviewed inputs.
    adjudications = contract.get('reviewed_invalid_inputs', [])
    if adjudications:
        result['coverage'] = 'partial'
        identity = digest(stdin)
        for item in adjudications:
            if item['stdin_sha256'] == identity:
                result['violations'].append({'reason': 'reviewed_invalid_input',
                    'detail': item['reason'], 'stdin_sha256': identity,
                    'review_sha256': item['review_sha256'], 'evidence': item['evidence']})
    return result


def _validate_multicase(contract, tokens):
    """Explicit counted cases; no scope or input position is inherited between cases."""
    result = {'coverage': contract.get('coverage', 'partial'), 'violations': []}
    header = _validate_flat_tokens(contract['header'], tokens, require_end=False)
    result['violations'].extend(header['violations'])
    cursor = header.get('_cursor', 0)
    count = header.get('_scalars', {}).get(contract['count_var'])
    if not header.get('_parsed_all') or count is None or count < 0:
        result['coverage'] = 'partial'
        return result
    if header['coverage'] != 'complete':
        result['coverage'] = 'partial'
    totals = [0] * len(contract.get('case_sums', []))
    for case_index in range(count):
        # A zero-token or ambiguous case must not allow a huge untrusted count
        # to spin, nor cause the next case's tokens to be interpreted blindly.
        case = _validate_flat_tokens(contract['case'], tokens, start=cursor, require_end=False)
        result['violations'].extend(dict(v, case_index=case_index) for v in case['violations'])
        next_cursor = case.get('_cursor', cursor)
        if not case.get('_parsed_all') or next_cursor <= cursor:
            result['coverage'] = 'partial'
            return result
        cursor = next_cursor
        if case['coverage'] != 'complete':
            result['coverage'] = 'partial'
        for i, rule in enumerate(contract.get('case_sums', [])):
            value = _eval_linear(rule['expression'], case['_scalars'])
            if value is None or totals[i] is None:
                totals[i] = None
                result['coverage'] = 'partial'
            else:
                totals[i] += value
    for total, rule in zip(totals, contract.get('case_sums', [])):
        if total is None:
            continue
        failed = total >= rule['limit'] if rule.get('operator') == '<' else total > rule['limit']
        if failed:
            result['violations'].append({'reason': 'case_sum_violated', 'rule_id': rule['id'],
                                         'evidence': rule['evidence'], 'value': total, 'limit': rule['limit']})
    if contract.get('layout_complete') and cursor < len(tokens):
        result['violations'].append({'reason': 'trailing_input_tokens', 'rule_id': 'input_layout',
                                     'evidence': contract.get('evidence', []), 'token_index': cursor,
                                     'value': tokens[cursor]})
    return result


def _validate_flat_tokens(contract: dict, tokens: list[str], start=0, require_end=True) -> dict:
    """Return only violations proven by compiled rules and their evidence."""
    coverage = contract.get("coverage", "unsupported") if isinstance(contract, dict) else "unsupported"
    result = {"coverage": coverage, "violations": []}
    if coverage == "unsupported":
        return result
    # Frozen review may prove the whole token layout while semantic guarantees remain partial.
    layout_complete = (contract.get("layout_complete") is True if "layout_complete" in contract
                       else coverage == "complete")
    operations = contract.get("operations", [])
    if not operations:
        return result

    cursor = start
    scalar_values: dict[str, int] = {}
    string_values: dict[str, str] = {}
    arrays: dict[str, list[int]] = defaultdict(list)
    row_values: dict[str, list[list[int]]] = defaultdict(list)
    parsed_all = True
    pending_bounds = []
    runtime_coverage = "partial" if coverage == "complete" and not layout_complete else coverage
    tokenization_blocked = False

    def consume_field(field: dict, row: list[int] | None = None,
                      row_context: dict | None = None) -> int | str | None:
        nonlocal cursor, parsed_all, runtime_coverage, tokenization_blocked
        if tokenization_blocked:
            parsed_all = False
            return None
        field_type = field.get("kind")
        bound_values = scalar_values | (row_context or {})
        if (field_type == "string"
                and not _string_field_has_token_safe_domain(field,bound_values)):
            runtime_coverage = "partial"
            parsed_all = False
            tokenization_blocked = True
            return None
        if cursor >= len(tokens):
            if layout_complete:
                result["violations"].append({"reason": "missing_input_token", "rule_id": field.get("name", "field"),
                                             "evidence": field.get("evidence", []), "token_index": cursor})
            parsed_all = False
            tokenization_blocked = True
            return None
        token_index = cursor
        token = tokens[cursor]
        cursor += 1
        if field_type == "int":
            if not re.fullmatch(r"[+-]?[0-9]+", token):
                result["violations"].append({"reason": "expected_integer", "rule_id": field.get("name", "field"),
                                             "evidence": field.get("evidence", []), "token_index": token_index,
                                             "value": token})
                parsed_all = False
                return None
            value = _integer_value(token)
            if row_context is not None:
                row_context[field.get("name")] = value
            if value is None:
                parsed_all = False
                runtime_coverage = "partial"
                result["violations"].extend(_huge_integer_violations(field, token, bound_values, token_index))
                return None
            violations, pending = _check_bounds(field, value, bound_values, token_index, token)
            for item in pending:
                # A separate dictionary per row survives until forward fields
                # are parsed; None masks same-named global fields until known.
                item["row_context"] = row_context
            result["violations"].extend(violations)
            pending_bounds.extend(pending)
            if field.get("name") and not field["name"].startswith("field_"):
                if row is None:
                    scalar_values[field["name"]] = value
                else:
                    row.append(value)
                    arrays[field["name"]].append(value)
            return value
        if field_type == "string":
            properties = {
                'string_required_character_missing': bool(field.get('contains_all')) and not set(field['contains_all']) <= set(token),
                'string_not_palindrome': field.get('is_palindrome') is True and token != token[::-1],
                'string_prefix_mismatch': bool(field.get('starts_with')) and not token.startswith(field['starts_with']),
                'string_suffix_mismatch': bool(field.get('ends_with')) and not token.endswith(field['ends_with']),
                'string_duplicate_character': field.get('distinct_characters') is True and len(set(token)) != len(token),
                'string_forbidden_prefix': any(token.startswith(p) for p in field.get('forbidden_prefixes',[])),
                'string_character_count_out_of_range': any(
                    token.count(c)<bounds.get('min',0) or ('max' in bounds and token.count(c)>bounds['max'])
                    for c,bounds in field.get('character_counts',{}).items()),
                'string_character_count_modulus_violation': any(
                    token.count(c)%divisor for c,divisor in field.get('character_count_moduli',{}).items()),
            }
            if field.get('alphabet_prefix_length_expr') is not None:
                alphabet_length = _eval_linear(field['alphabet_prefix_length_expr'],bound_values)
                if alphabet_length is None:
                    runtime_coverage = 'partial'
                else:
                    allowed = field['alphabet'][:max(0,alphabet_length)]
                    properties['string_alphabet_prefix_violation'] = any(c not in allowed for c in token)
            if field.get('subsequence_of'):
                parent = string_values.get(field['subsequence_of'])
                if parent is None:
                    runtime_coverage = 'partial'
                else:
                    iterator = iter(parent)
                    properties['string_not_subsequence'] = not all(any(x == c for x in iterator) for c in token)
            for reason, failed in properties.items():
                if failed:
                    result['violations'].append({'reason':reason,'rule_id':field.get('name','string'),
                        'evidence':field.get('evidence',[]),'token_index':token_index})
            if field.get("alphabet") is not None and any(c not in field["alphabet"] for c in token):
                result["violations"].append({"reason":"string_alphabet_violation",
                    "rule_id":field.get("name","string"),"evidence":field.get("evidence",[]),
                    "token_index":token_index})
            for key in ("min_length_expr","max_length_expr"):
                if field.get(key) is None:
                    continue
                bound = _eval_linear(field[key],bound_values)
                if bound is None:
                    runtime_coverage = "partial"
                elif (key=="min_length_expr" and len(token)<bound) or (key=="max_length_expr" and len(token)>bound):
                    result["violations"].append({"reason":"string_length_out_of_range",
                        "rule_id":field.get("name","string"),"evidence":field.get("evidence",[]),
                        "token_index":token_index,"length":len(token),"bound":bound})
            if field.get("length_expr") is not None:
                expected_length = _eval_linear(field["length_expr"], bound_values)
                if expected_length is None:
                    runtime_coverage = "partial"
                elif len(token) != expected_length:
                    result["violations"].append({"reason": "string_length_mismatch", "rule_id": field.get("name", "string"),
                                                 "evidence": field.get("evidence", []), "token_index": token_index,
                                                 "value": token, "expected_length": expected_length})
            if row_context is None:
                if field.get('length_binding'):
                    scalar_values[field['length_binding']] = len(token)
                if field.get('name'):
                    string_values[field['name']] = token
            elif field.get('length_binding') or field.get('subsequence_of'):
                # Cross-row string relations need an explicit row grammar.
                runtime_coverage = 'partial'
            return token
        parsed_all = False
        return None

    operation_rows: dict[str, list[list[int]]] = defaultdict(list)
    for operation in operations:
        if tokenization_blocked:
            break
        if operation.get("kind") == "repeat":
            count = _eval_linear(operation.get("count_expr", {}), scalar_values)
            if count is None or count < 0:
                parsed_all = False
                runtime_coverage = "partial" if coverage != "unsupported" else coverage
                break
            required = count * max(1, len(operation.get("fields", [])))
            remaining = len(tokens) - cursor
            if required > remaining and not any(f.get("kind") == "string" for f in operation.get("fields", [])):
                if layout_complete:
                    field = (operation.get("fields") or [{}])[0]
                    result["violations"].append({"reason": "missing_input_token", "rule_id": field.get("name", operation["id"]),
                                                 "evidence": field.get("evidence", operation.get("evidence", [])),
                                                 "token_index": len(tokens)})
                else:
                    runtime_coverage = "partial"
                parsed_all = False
                break
            for _ in range(count):
                row = []
                context = ({field.get("name"): None for field in operation.get("fields", [])}
                           if operation.get("rows") or operation.get("array") else None)
                for field in operation.get("fields", []):
                    consume_field(field, row if context is not None else None, context)
                    if tokenization_blocked:
                        break
                if tokenization_blocked:
                    break
                if context is not None:
                    operation_rows[operation["id"]].append(row)
                elif len(operation.get("fields", [])) == 1:
                    field_name = operation["fields"][0].get("name")
                    arrays[field_name] = arrays.get(field_name, [])
        else:
            values = []
            for field in operation.get("fields", []):
                values.append(consume_field(field))
                if tokenization_blocked:
                    break
            operation_rows[operation['id']].append(values)

    for pending in pending_bounds:
        field = pending["field"]
        bound_values = scalar_values | (pending.get("row_context") or {})
        violations, still_pending = _check_bounds(field, pending["value"], bound_values,
                                                   pending["token_index"], pending["token"])
        if still_pending:
            runtime_coverage = "partial"
        result["violations"].extend(violations)

    if require_end and layout_complete and parsed_all and cursor < len(tokens):
        result["violations"].append({"reason": "trailing_input_tokens", "rule_id": "input_layout",
                                     "evidence": _union_evidence(operations), "token_index": cursor,
                                     "value": tokens[cursor]})

    for constraint in contract.get("constraints", []):
        if constraint.get("kind") not in {"upper_bound","product_upper_bound"}:
            continue
        if constraint['kind']=='product_upper_bound':
            value=1
            for factor in constraint['factors']:
                number=_eval_linear(factor,scalar_values)
                if number is None:
                    value=None
                    break
                value*=number
        else:value = _eval_linear(constraint.get("expression", {}), scalar_values)
        if value is None:
            runtime_coverage = "partial"
            continue
        limit = constraint.get("limit")
        operator = constraint.get("operator", "≤")
        failed = value > limit if operator in {"≤", "<=", "\\le", "\\leq"} else value >= limit
        if failed:
            result["violations"].append({"reason": "constraint_violated", "rule_id": constraint["id"],
                                         "evidence": constraint["evidence"], "value": value, "limit": limit})

    if not parsed_all:
        runtime_coverage = "partial"
    for semantic in contract.get("semantics", []):
        # Unknown or incomplete parsing is not evidence of an invalid graph or
        # permutation. Conservatively defer all semantic checks in that case.
        if not parsed_all:
            continue
        operation_id = semantic.get("operation_id")
        kind = semantic.get('kind')
        if kind=='row_linear_upper_bound':
            op=next((item for item in operations if item['id']==operation_id),None)
            rows=operation_rows.get(operation_id,[])
            names=[f.get('name') for f in op['fields']] if op else []
            count=1 if op and op['kind']=='fields' else _eval_linear(op.get('count_expr',{}),scalar_values) if op else None
            if count is None or len(rows)!=count or any(len(row)!=len(names) for row in rows):
                runtime_coverage='partial'
                continue
            values=[_eval_linear(semantic['expression'],scalar_values|dict(zip(names,row))) for row in rows]
            if any(value is None for value in values):
                runtime_coverage='partial'
                continue
            if any(value>semantic['limit'] for value in values):
                result['violations'].append({'reason':'row_linear_upper_bound_violated','rule_id':semantic['id'],
                    'operation_id':operation_id,'evidence':semantic['evidence']})
            continue
        if kind in {'row_pairwise_neq','unique_tuple_rows','row_gcd_one',
                    'array_sum_upper_bound','array_sum_lower_bound','array_sum_equals',
                    'aggregate_gcd_one','multiple_of_constant','row_sum_lower_bound',
                    'row_sum_upper_bound','unique_unordered_pair_rows','row_tuple_neq'}:
            op = next((item for item in operations if item['id'] == operation_id), None)
            names = [field.get('name') for field in op.get('fields', [])] if op else []
            wanted = semantic.get('field_names', [semantic.get('field_name')])
            if kind=='row_tuple_neq':
                wanted=[n for group in semantic['field_groups'] for n in group]
            count = _eval_linear(op.get('count_expr', {}), scalar_values) if op else None
            if op and op['kind'] == 'fields':
                count = 1
            rows = operation_rows.get(operation_id, [])
            if (op is None or count is None or len(rows) != count or any(n not in names for n in wanted)
                    or any(len(row) != len(names) for row in rows)):
                runtime_coverage = 'partial'
                continue
            tuples = [tuple(row[names.index(n)] for n in wanted) for row in rows]
            if kind == 'row_pairwise_neq':
                failed = any(len(set(row)) != len(row) for row in tuples)
            elif kind == 'unique_tuple_rows':
                failed = len(set(tuples)) != len(tuples)
            elif kind == 'unique_unordered_pair_rows':
                failed = len({tuple(sorted(row)) for row in tuples}) != len(tuples)
            elif kind == 'row_tuple_neq':
                width=len(semantic['field_groups'][0])
                failed=any(row[:width]==row[width:] for row in tuples)
            elif kind == 'row_gcd_one':
                failed = any(gcd(*row) != 1 for row in tuples)
            elif kind == 'multiple_of_constant':
                failed = any(row[0] % semantic['divisor'] for row in tuples)
            elif kind == 'aggregate_gcd_one':
                accumulated = 0
                for row in tuples:
                    accumulated = gcd(accumulated, row[0])
                failed = accumulated != 1
            else:
                maximum = _eval_linear(semantic['expression'], scalar_values)
                if maximum is None:
                    runtime_coverage = 'partial'
                    continue
                if kind.startswith('row_sum_'):
                    failed=any(sum(row)<maximum if kind=='row_sum_lower_bound' else sum(row)>maximum for row in tuples)
                else:
                    total = sum(row[0] for row in tuples)
                    failed = (total > maximum if kind == 'array_sum_upper_bound' else
                              total < maximum if kind == 'array_sum_lower_bound' else total != maximum)
            if failed:
                result['violations'].append({'reason': kind+'_violated', 'rule_id': semantic['id'],
                    'operation_id': operation_id, 'evidence': semantic['evidence'], 'row_count': len(rows)})
            continue
        if semantic.get("kind") in {"all_distinct", "ordered", "excludes_value"}:
            op = next((item for item in operations if item["id"] == operation_id), None)
            names = [field.get("name") for field in op.get("fields", [])] if op else []
            name = semantic.get("field_name")
            count = _eval_linear(op.get("count_expr", {}), scalar_values) if op else None
            rows = operation_rows.get(operation_id, [])
            if (op is None or count is None or name not in names or len(rows) != count
                    or any(len(row) != len(names) for row in rows)):
                runtime_coverage = "partial"
                continue
            values = [row[names.index(name)] for row in rows]
            kind = semantic["kind"]
            if kind == "all_distinct":
                failed = len(set(values)) != len(values)
            elif kind == "ordered":
                comparisons = {"lt": lambda a,b:a<b, "le": lambda a,b:a<=b,
                               "gt": lambda a,b:a>b, "ge": lambda a,b:a>=b}
                compare = comparisons.get(semantic.get("operator"))
                if compare is None:
                    runtime_coverage = "partial"
                    continue
                failed = any(not compare(a,b) for a,b in zip(values,values[1:]))
            else:
                excluded = _eval_linear(semantic.get("expression", {}), scalar_values)
                if excluded is None:
                    runtime_coverage = "partial"
                    continue
                failed = excluded in values
            if failed:
                result["violations"].append({"reason": "array_"+kind+"_violated",
                    "rule_id": semantic["id"], "evidence": semantic["evidence"],
                    "operation_id": operation_id, "value_count": len(values)})
            continue
        if semantic.get("kind") == "permutation":
            name = semantic.get("field_name")
            expected = scalar_values.get(semantic.get("length_var", "n"))
            op = next((item for item in operations if item["id"] == operation_id), None)
            count = _eval_linear(op.get("count_expr", {}), scalar_values) if op else None
            expected = count if count is not None else expected
            names = [field.get("name") for field in op.get("fields", [])] if op else []
            rows = operation_rows.get(operation_id, [])
            if (op is None or expected is None or name not in names or len(rows) != expected
                    or any(len(row) != len(names) for row in rows)):
                runtime_coverage = "partial"
                continue
            values = [row[names.index(name)] for row in rows]
            if len(values) != expected or (values and (min(values) != 1 or max(values) != expected)) or len(set(values)) != len(values):
                result["violations"].append({"reason": "not_a_permutation", "rule_id": semantic["id"],
                                             "evidence": semantic["evidence"], "value_count": len(values),
                                             "expected_count": expected})
        elif semantic.get("kind") == "tree":
            op = next((item for item in operations if item["id"] == operation_id), None)
            node_count = scalar_values.get(semantic.get("node_count_var"))
            if op is None or node_count is None:
                runtime_coverage = "partial"
                continue
            edges = [tuple(row[:2]) for row in operation_rows.get(operation_id, []) if len(row) >= 2]
            if node_count is not None and not _tree_is_valid(node_count, edges):
                result["violations"].append({"reason": "not_a_tree", "rule_id": semantic["id"],
                                             "evidence": semantic["evidence"], "edge_count": len(edges),
                                             "node_count": node_count})

    result["coverage"] = runtime_coverage
    result.update(_cursor=cursor, _scalars=scalar_values, _parsed_all=parsed_all)
    return result
