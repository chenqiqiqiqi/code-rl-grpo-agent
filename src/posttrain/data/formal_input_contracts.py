"""Strict compiler for simple AtCoder-style raw input-format tables.

This module complements ``constraint_engine`` when a statement gives a compact
symbol table (for example ``N M`` and ``A_1 A_2 ... A_N``) instead of prose
describing each input line. Symbols are treated as integers only when the
statement explicitly says so. Unrecognized rows stop parsing and lower
coverage; numeric bounds alone never establish an integer type.
"""

from __future__ import annotations

import re

from posttrain.data.constraint_engine import (
    SCHEMA_VERSION,
    _evidence,
    _normalize_name,
    _parse_linear,
    _range_rules,
    _sections,
    _semantic_checks,
)


_FORMAT_RE = re.compile(r"\bformat\s*:", re.IGNORECASE)
_SIMPLE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_INDEXED_NAME_RE = re.compile(r"(?P<base>[A-Za-z][A-Za-z0-9]*)_\{?(?P<index>[A-Za-z][A-Za-z0-9]*|\d+)\}?\Z")
_ELLIPSIS = {"...", "…", "⋯", r"\cdots", r"\ldots", r"\dots", r"\vdots"}
_WORDS = {
    "and", "or", "input", "output", "format", "from", "to", "of", "the", "where", "which",
    # These tokens can occur in type phrases but are not input-variable names.
    # Excluding them prevents patterns like "integer sequence of length N"
    # from typing phantom fields named ``sequence`` or ``of``.
    "sequence", "sequences", "array", "arrays", "list", "lists", "integer", "integers",
    "length", "size", "value", "values", "element", "elements", "number", "numbers",
}
_UNHANDLED_RELATION_RE = re.compile(r"(?<![<>=])=(?!=)|==|!=|≠|≢|∑|Σ|\\sum\b", re.IGNORECASE)
_UNHANDLED_GUARANTEE_RE = re.compile(
    r"\b(?:if|when|unless|otherwise|provided|such\s+that|for\s+every|for\s+each|"
    r"there\s+exists|exactly|sum|product|total|divisible|multiple|sorted|increasing|"
    r"decreasing|distinct|permutation|tree|connected|even|odd|prime|positive|negative|"
    r"nonzero|at\s+least|at\s+most|equal(?:s)?|different|unique|guaranteed|must|"
    r"greater\s+than|less\s+than|digit|character|substring|contain(?:s|ing)?|"
    r"consist(?:s|ing)?\s+of|made\s+of|one\s+of)\b",
    re.IGNORECASE,
)


def _clean_symbol(token: str) -> str:
    value = token.strip().strip("$` ")
    value = value.strip(",;:()[]")
    value = value.replace("\\{", "{").replace("\\}", "}")
    return value


def _line_records(statement: str, start: int, end: int) -> list[dict]:
    records = []
    cursor = start
    for raw in statement[start:end].splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        left = len(content) - len(content.lstrip())
        right = len(content.rstrip())
        if right > left:
            records.append({"start": cursor + left, "end": cursor + right, "text": content[left:right]})
        cursor += len(raw)
    return records


def _find_format_table(statement: str, input_span: tuple[int, int]) -> tuple[list[dict] | None, dict | None]:
    section_start, section_end = input_span
    text = statement[section_start:section_end]
    markers = list(_FORMAT_RE.finditer(text))
    if not markers:
        return None, None
    marker = markers[0]
    format_start = section_start + marker.start()
    format_end = section_start + marker.end()
    intro = _evidence(statement, format_start, format_end, "input_format")
    return _line_records(statement, format_end, section_end), intro


def _parse_raw_row(text: str) -> dict | None:
    tokens = [_clean_symbol(token) for token in text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return None

    if len(tokens) <= 20 and all(_SIMPLE_NAME_RE.fullmatch(token) and token.lower() not in _WORDS for token in tokens):
        # A raw scalar row is a sequence of explicit, unindexed symbols.
        if len(set(tokens)) != len(tokens):
            return None
        return {"kind": "scalars", "names": tokens}

    if len(tokens) != 4 or tokens[2] not in _ELLIPSIS:
        return None
    first = _INDEXED_NAME_RE.fullmatch(tokens[0])
    second = _INDEXED_NAME_RE.fullmatch(tokens[1])
    last = _INDEXED_NAME_RE.fullmatch(tokens[3])
    if not first or not second or not last:
        return None
    if first.group("base") != second.group("base") or first.group("base") != last.group("base"):
        return None
    if first.group("index") != "1" or second.group("index") != "2":
        return None
    terminal = last.group("index")
    if not terminal.isalpha():
        return None
    return {"kind": "array", "base": first.group("base"), "count_var": terminal}


def _scope_spans(input_span: tuple[int, int], constraint_span: tuple[int, int] | None) -> list[tuple[int, int, str]]:
    spans = [(input_span[0], input_span[1], "input_constraint")]
    if constraint_span is not None:
        spans.append((constraint_span[0], constraint_span[1], "constraint_section"))
    return spans


def _integer_type_evidence(statement: str, input_span: tuple[int, int],
                           constraint_span: tuple[int, int] | None) -> tuple[list[dict], set[str], set[str], dict[str, list[dict]]]:
    """Find only explicit integer-type statements, without using range syntax."""
    spans: list[tuple[int, int, str]] = [(0, input_span[0], "problem_statement"),
                                         (input_span[0], input_span[1], "input_constraint")]
    if constraint_span is not None:
        spans.append((constraint_span[0], constraint_span[1], "constraint_section"))

    global_evidence = []
    scalar_names: set[str] = set()
    array_bases: set[str] = set()
    typed_evidence: dict[str, list[dict]] = {}

    def record_type(name: str, evidence: dict, *, array: bool) -> None:
        normalized = _normalize_name(name, indexed=array)
        if normalized.lower() in _WORDS:
            return
        typed_evidence.setdefault(normalized, []).append(evidence)
        if array:
            array_bases.add(re.sub(r"_i$", "", normalized))
        else:
            scalar_names.add(normalized)
    global_patterns = [
        re.compile(r"\ball\s+(?:(?:of\s+)?the\s+)?(?:values?|numbers?|elements?)\s+(?:in|of)\s+(?:the\s+)?input\s+(?:are|consist\s+of)\s+integers?\b", re.IGNORECASE),
        re.compile(r"\ball\s+(?:input\s+)?values\s+are\s+integers?\b", re.IGNORECASE),
        re.compile(r"\binput\s+(?:data|values?)\s+(?:are|consist\s+of)\s+integers?\b", re.IGNORECASE),
        re.compile(r"\bthe\s+input\s+consists\s+of\s+integers?\b", re.IGNORECASE),
    ]
    for start, end, role in spans:
        local = statement[start:end]
        for pattern in global_patterns:
            for match in pattern.finditer(local):
                global_evidence.append(_evidence(statement, start + match.start(), start + match.end(), role))

        # Singular declarations such as "N is a positive integer" or
        # "integer N" can establish the type of that named scalar only.
        scalar_patterns = [
            re.compile(r"\b(?P<name>[A-Za-z][A-Za-z0-9]*)\s+(?:is|are)\s+(?:(?:a|an)\s+)?(?:(?:positive|negative|nonnegative|non-negative|nonpositive|non-positive)\s+)?integers?\b", re.IGNORECASE),
            re.compile(r"\b(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integer\s+(?P<name>[A-Za-z][A-Za-z0-9]*)\b", re.IGNORECASE),
        ]
        for pattern in scalar_patterns:
            for match in pattern.finditer(local):
                record_type(match.group("name"), _evidence(statement, start + match.start(), start + match.end(), role), array=False)

        plural_scalar = re.compile(
            r"(?P<names>[A-Za-z][A-Za-z0-9]*(?:\s*(?:,|\band\b)\s*[A-Za-z][A-Za-z0-9]*)*)"
            r"\s+(?:are|are all)\s+(?:(?:positive|negative|nonnegative|non-negative|nonpositive|non-positive)\s+)?integers?\b",
            re.IGNORECASE,
        )
        for match in plural_scalar.finditer(local):
            evidence = _evidence(statement, start + match.start(), start + match.end(), role)
            for name in re.findall(r"[A-Za-z][A-Za-z0-9]*", match.group("names")):
                if name.lower() not in _WORDS:
                    record_type(name, evidence, array=False)

        # Explicitly typed integer sequences bind their element family, e.g.
        # "a sequence of N positive integers A_1, ..., A_N" or "A is a
        # sequence of integers".
        array_patterns = [
            re.compile(r"\b(?:sequence|array|list)\s+(?:of\s+)?(?:\w+\s+)?(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integers?\s+(?P<name>[A-Za-z][A-Za-z0-9]*_\{?[A-Za-z0-9]+\}?)", re.IGNORECASE),
            re.compile(r"\b(?P<name>[A-Za-z][A-Za-z0-9]*)\s+(?:is|are)\s+(?:a\s+)?(?:sequence|array|list)\s+of\s+(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integers?\b", re.IGNORECASE),
            re.compile(r"\b(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integers?\s+(?P<name>[A-Za-z][A-Za-z0-9]*_\{?[A-Za-z0-9]+\}?)", re.IGNORECASE),
            re.compile(r"\b(?P<name>[A-Za-z][A-Za-z0-9]*)\s+(?:is|are)\s+(?:(?:a|an)\s+)?(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integer\s+(?:sequence|array|list)\b", re.IGNORECASE),
            re.compile(r"\b(?:integer|positive\s+integer|negative\s+integer)\s+(?:sequence|array|list)\s+(?P<name>[A-Za-z][A-Za-z0-9]*)(?!\s*(?:of|in|with)\b)", re.IGNORECASE),
            re.compile(r"\b(?P<name>[A-Za-z][A-Za-z0-9]*_\{?[A-Za-z0-9]+\}?)\s+(?:are|are all)\s+(?:positive\s+|negative\s+|nonnegative\s+|non-negative\s+)?integers?\b", re.IGNORECASE),
        ]
        for pattern in array_patterns:
            for match in pattern.finditer(local):
                record_type(match.group("name"), _evidence(statement, start + match.start(), start + match.end(), role), array=True)

    # Deduplicate exact span records while preserving source order.
    seen = set()
    evidence = []
    for item in global_evidence:
        key = (item["start"], item["end"])
        if key not in seen:
            seen.add(key)
            evidence.append(item)
    return evidence, scalar_names, array_bases, typed_evidence


def _known_integer(name: str, *, array: bool, global_evidence: list[dict],
                   scalar_names: set[str], array_bases: set[str]) -> bool:
    if global_evidence or name in scalar_names:
        return True
    base = re.sub(r"_i$", "", name)
    return array and base in array_bases


def _field(name: str, *, array: bool, statement: str, row_evidence: dict,
           global_type_evidence: list[dict], scalar_types: set[str], array_types: set[str],
           typed_evidence: dict[str, list[dict]], range_map: dict) -> dict:
    normalized = _normalize_name(name, indexed=array)
    integer_typed = _known_integer(normalized, array=array, global_evidence=global_type_evidence,
                                   scalar_names=scalar_types, array_bases=array_types)
    evidence = [row_evidence]
    bounds = []
    if integer_typed:
        if global_type_evidence:
            evidence.extend(global_type_evidence)
        else:
            evidence.extend(typed_evidence.get(normalized, []))
            if array:
                evidence.extend(typed_evidence.get(re.sub(r"_i$", "", normalized), []))
        for rule in range_map.get(normalized, []):
            # A rejection must carry evidence for the numeric bound and for
            # why this raw table slot is an integer field. Keep source slices
            # intact so callers can verify every item against the statement.
            bound_evidence = list(rule["evidence"]) + [row_evidence]
            if global_type_evidence:
                bound_evidence.extend(global_type_evidence)
            else:
                bound_evidence.extend(typed_evidence.get(normalized, []))
                if array:
                    bound_evidence.extend(typed_evidence.get(re.sub(r"_i$", "", normalized), []))
            unique = {}
            for item in bound_evidence:
                unique[(item["start"], item["end"])] = item
            bound = {"min_expr": rule["min_expr"], "max_expr": rule["max_expr"],
                     "evidence": list(unique.values())}
            bounds.append(bound)
            evidence.extend(bound["evidence"])
    return {"kind": "int" if integer_typed else "opaque", "name": normalized,
            "bounds": bounds, "evidence": evidence}


def _unsupported(statement: str, reason: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "coverage": "unsupported", "operations": [],
            "constraints": [], "semantics": [], "limitations": [reason],
            "compiler": "formal_v1", "sound_for_deletion": False}


def compile_formal_contract(statement: str) -> dict:
    """Compile a strict raw-format-table subset to the constraint-engine schema."""
    if not isinstance(statement, str) or not statement:
        return _unsupported(statement if isinstance(statement, str) else "", "missing_statement")
    input_span, constraint_span = _sections(statement)
    if input_span is None:
        return _unsupported(statement, "input_section_not_found")

    records, intro_evidence = _find_format_table(statement, input_span)
    if records is None or intro_evidence is None:
        return _unsupported(statement, "raw_format_table_not_found")

    type_evidence, scalar_types, array_types, typed_evidence = _integer_type_evidence(statement, input_span, constraint_span)
    range_map, _recognized, unresolved_inequality = _range_rules(statement, _scope_spans(input_span, constraint_span))
    cross_rules = range_map.pop("__cross__", [])

    operations: list[dict] = []
    parsed_scalar_names: set[str] = set()
    limitations: list[str] = []
    halted = False
    nonempty_records = 0
    for record_index, record in enumerate(records):
        if halted:
            break
        parsed = _parse_raw_row(record["text"])
        if parsed is None:
            halted = True
            limitations.append("unrecognized_input_format_row")
            break
        nonempty_records += 1
        row_evidence = _evidence(statement, record["start"], record["end"], "input_format")

        if parsed["kind"] == "scalars":
            fields = []
            stop_after_row = False
            for symbol in parsed["names"]:
                field = _field(symbol, array=False, statement=statement, row_evidence=row_evidence,
                               global_type_evidence=type_evidence, scalar_types=scalar_types,
                               array_types=array_types, typed_evidence=typed_evidence, range_map=range_map)
                fields.append(field)
                if field["kind"] == "int":
                    parsed_scalar_names.add(field["name"])
                else:
                    # Consume this row's explicit slots as opaque values, then
                    # stop. Later rows cannot be bound after an unknown type.
                    stop_after_row = True
                    break
            operations.append({"id": f"op{len(operations)}", "kind": "fields", "fields": fields,
                               "evidence": [row_evidence, intro_evidence]})
            if stop_after_row:
                halted = True
                limitations.append("integer_type_not_established_for_all_fields")
            continue

        count_var = parsed["count_var"]
        count_expression = _parse_linear(count_var)
        array_name = _normalize_name(parsed["base"] + "_i", indexed=True)
        if count_expression is None or count_var not in parsed_scalar_names:
            halted = True
            limitations.append("array_length_not_bound_to_prior_integer_scalar")
            break
        field = _field(array_name, array=True, statement=statement, row_evidence=row_evidence,
                       global_type_evidence=type_evidence, scalar_types=scalar_types,
                       array_types=array_types, typed_evidence=typed_evidence, range_map=range_map)
        operation = {"id": f"op{len(operations)}", "kind": "repeat", "fields": [field],
                     "count_expr": count_expression, "array": True, "distinct": False,
                     "evidence": [row_evidence, intro_evidence]}
        operations.append(operation)
        if field["kind"] != "int":
            halted = True
            limitations.append("integer_type_not_established_for_all_fields")

    if not operations or nonempty_records == 0:
        return _unsupported(statement, "no_recognized_raw_format_rows")

    # Report bounds that were present but could not be bound to parsed fields.
    attached_names = {field["name"] for operation in operations for field in operation["fields"]}
    unattached = [name for name, items in range_map.items() if items and name not in attached_names]
    if unattached:
        limitations.append("some_bounds_not_bound_to_raw_fields")
    scalar_field_names = {
        field["name"] for operation in operations if operation.get("kind") == "fields"
        for field in operation.get("fields", []) if field.get("kind") == "int"
    }
    unresolved_bound_reference = False
    for operation in operations:
        for field in operation.get("fields", []):
            for bound in field.get("bounds", []):
                for side in (bound.get("min_expr"), bound.get("max_expr")):
                    if side is not None and not set(side.get("terms", {})).issubset(scalar_field_names):
                        unresolved_bound_reference = True
    for rule in cross_rules:
        if not set(rule.get("expression", {}).get("terms", {})).issubset(scalar_field_names):
            unresolved_bound_reference = True
    if unresolved_bound_reference:
        limitations.append("bound_references_non_scalar_or_unparsed_field")
    # The range helper may include a Markdown list marker in the numeric token
    # ("- 1 <= N"), where the marker is not a minus sign. Until the helper
    # separates those cases, such contracts can diagnose conservatively but
    # cannot claim full input-domain coverage.
    bullet_bound_ambiguity = any(
        item.get("role") == "constraint_section"
        and re.match(r"^\s*[-*+]\s+\d", item.get("text", ""))
        for operation in operations for field in operation.get("fields", [])
        for bound in field.get("bounds", []) for item in bound.get("evidence", [])
    )
    if bullet_bound_ambiguity:
        limitations.append("possible_list_marker_bound_ambiguity")
    if not type_evidence and not scalar_types and not array_types:
        limitations.append("input_integer_type_not_established")
    if unresolved_inequality:
        limitations.append("some_inequalities_not_compiled")

    semantics, unresolved_semantics = _semantic_checks(statement, input_span, constraint_span, operations)
    if unresolved_semantics:
        limitations.append("some_semantic_constraints_not_compiled")
    # Capture additional common guarantees not covered by the frozen helper.
    context_spans = [(0, input_span[0])]
    context_spans.extend((start, end) for start, end, _role in _scope_spans(input_span, constraint_span))
    extra_semantic = re.compile(
        r"\b(?:multiple\s+of|divisible\s+by|sorted|in\s+increasing\s+order|"
        r"non[- ]decreasing|non[- ]increasing|pairwise\s+different|all\s+different|"
        r"all\s+distinct|at\s+least\s+once)\b", re.IGNORECASE,
    )
    if any(extra_semantic.search(statement[start:end]) for start, end in context_spans):
        limitations.append("some_semantic_constraints_not_compiled")
    # The range helper intentionally handles only proven interval and simple
    # linear upper-bound forms. Equality, conditional, and broader mathematical
    # requirements remain outside this formal-table subset. Any such cue keeps
    # the contract usable for diagnostics but untrusted for removal.
    if any(_UNHANDLED_RELATION_RE.search(statement[start:end]) for start, end in context_spans):
        limitations.append("uncompiled_relation_or_formula")
    if any(_UNHANDLED_GUARANTEE_RE.search(statement[start:end]) for start, end in context_spans):
        limitations.append("uncompiled_domain_or_semantic_guarantee")

    input_fields = [field for operation in operations for field in operation["fields"]]
    unbounded = [field for field in input_fields
                 if field.get("kind") != "int" or not field.get("bounds")]
    incomplete_interval = [
        field for field in input_fields if field.get("kind") == "int" and (
            not any(bound.get("min_expr") is not None for bound in field.get("bounds", []))
            or not any(bound.get("max_expr") is not None for bound in field.get("bounds", []))
        )
    ]
    if any(field.get("kind") != "int" for operation in operations for field in operation["fields"]):
        limitations.append("integer_type_not_established_for_all_fields")
    if any(not field.get("bounds") for operation in operations for field in operation["fields"]
           if field.get("kind") == "int"):
        limitations.append("some_input_fields_have_no_compiled_range")
    if incomplete_interval:
        limitations.append("some_input_fields_lack_a_compiled_lower_or_upper_bound")
    if halted:
        limitations.append("input_table_parsing_stopped_at_unhandled_row")

    # Only close a contract when every input row, field type, numeric rule and
    # detected semantic guarantee is represented.
    complete = bool(operations) and not halted and not unattached and not unresolved_inequality and \
        not unresolved_semantics and not unbounded and not incomplete_interval and not any(
            item in limitations for item in ("some_semantic_constraints_not_compiled", "input_integer_type_not_established",
                                             "uncompiled_relation_or_formula", "uncompiled_domain_or_semantic_guarantee",
                                             "bound_references_non_scalar_or_unparsed_field",
                                             "possible_list_marker_bound_ambiguity")
        )
    coverage = "complete" if complete else "partial"

    constraints = []
    for operation in operations:
        for field in operation["fields"]:
            for index, bound in enumerate(field.get("bounds", [])):
                constraints.append({"id": f"{operation['id']}:{field['name']}:range{index}",
                                    "kind": "range", "field_name": field["name"],
                                    "min_expr": bound["min_expr"], "max_expr": bound["max_expr"],
                                    "evidence": bound["evidence"]})
    for index, rule in enumerate(cross_rules):
        constraints.append({"id": f"constraint:cross{index}", "kind": "upper_bound",
                            "expression": rule["expression"], "operator": rule["op"],
                            "limit": rule["limit"], "evidence": rule["evidence"]})

    return {"schema_version": SCHEMA_VERSION, "coverage": coverage, "operations": operations,
            "constraints": constraints, "semantics": semantics,
            "limitations": list(dict.fromkeys(limitations)), "compiler": "formal_v1",
            "sound_for_deletion": coverage == "complete"}
