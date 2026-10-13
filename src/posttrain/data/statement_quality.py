"""Conservative, statement-only quality triage for the training pool.

``no_confirmed_issue`` means this screen found no concrete defect. It is not a
claim that the task is complete, solvable, or validated against its tests.
"""
from __future__ import annotations

import re

from posttrain.data.eligibility import statement_issues


VERSION = "statement-quality-v1"

_SECTION = re.compile(
    r"(?im)^[ \t]*[-=#* \t]*(?P<name>input|output|constraints|examples?(?:[ \t]+\d+)?|"
    r"sample(?:[ \t]+(?:input|output|test(?:[ \t]+cases?)?))?|notes?|explanation)"
    r"[ \t]*:?[ \t]*[-=#* \t]*$"
)
_TASK_RELATION = re.compile(
    r"\b(?:find|determine|calculate|compute|decide|check(?:\s+(?:whether|if))?|"
    r"construct|build|generate|perform|replace|transform|convert|sort|reverse|"
    r"concatenate|insert|delete|remove|add|subtract|multiply|divide|sum|product|"
    r"maximum|minimum|average|mean|median|count|number\s+of|length\s+of|area|"
    r"perimeter|distance|difference|remainder|modulo|whether|otherwise|if|when|"
    r"called|defined\s+as|considered|such\s+that|iff|palindrome|prime|divisible|"
    r"equal|connected|reachable|result\s+of|value\s+of)\b",
    re.IGNORECASE,
)
_ARITHMETIC_OUTPUT = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_]*\s*(?:\+|-|\*|/|%|\bmod\b)\s*"
    r"(?:[A-Za-z][A-Za-z0-9_]*|\d+)"
)
_CHOICE_ONLY_OUTPUT = re.compile(
    r'''(?is)^\s*(?P<claim>(?:output|print|write)\s+(?:"[^"\n]+"'''
    r'''(?:\s+or\s+"[^"\n]+")+|'[^'\n]+'(?:\s+or\s+'[^'\n]+')+|'''
    r'''[A-Z][A-Z0-9_]*(?:\s+or\s+[A-Z][A-Z0-9_]*)+)(?:[.!])?)\s*$'''
)
_TYPE_ONLY_OUTPUT = re.compile(
    r"(?is)^\s*(?P<claim>(?:output|print|write)\s+(?:(?:an?|the)\s+)?"
    r"(?:single\s+)?(?:integer|number|real\s+number|string|line|word|character)(?:[.!])?)\s*$"
)
_BASIC_INPUT_SCHEMA = re.compile(
    r"(?is)^\s*(?:(?:the\s+)?input|(?:the\s+)?(?:first|second|third|only)\s+line)\s+"
    r"(?:contains|consists\s+of|is|has)\s+(?:a\s+)?"
    r"(?:(?:single|one|two|three|four|five|six|\d+)\s+)?"
    r"(?:(?:positive|nonnegative|non-negative|negative|nonzero|non-zero)\s+)*"
    r"(?:integer|number|real\s+number|string|word)s?\s+"
    r"\$?[A-Za-z][A-Za-z0-9_]*\$?(?:\s*,\s*\$?[A-Za-z][A-Za-z0-9_]*\$?)*"
    r"(?:\s*\([^)]*\))?"
    r"(?:\s*,?\s*separated\s+by\s+(?:a\s+)?(?:single\s+)?(?:space|spaces|newline|line))?"
    r"\s*[.!]?\s*$"
)
_BOUND_NUMBER = (
    r"-?(?:\d{1,3}(?:(?:,\d{3})|(?:[ \t\xa0]\d{3})|(?:[ \t\xa0]*\{,\}[ \t\xa0]*\d{3}))+|\d+)"
    r"(?![\w.^*/×·⋅'_])(?!,\d)(?!(?:[ \t\xa0]*(?:[\^*/+×·⋅]|\\(?:times|cdot))))"
)
_BOUND_OPERATOR = r"<=|>=|≤|≥|<|>|\\leq?|\\geq?"
_CHAINED_BOUNDS = re.compile(
    rf"(?<![\w.^,])(?P<left>{_BOUND_NUMBER})[ \t\xa0]*(?P<op1>{_BOUND_OPERATOR})[ \t\xa0]*"
    rf"(?P<var>[A-Za-z][A-Za-z0-9_]*)[ \t\xa0]*(?P<op2>{_BOUND_OPERATOR})[ \t\xa0]*"
    rf"(?P<right>{_BOUND_NUMBER})"
)
_VISUAL_REFERENCE = re.compile(
    r"\b(?:see|refer\s+to|use|read|look\s+at|shown\s+in|as\s+shown\s+in)\s+"
    r"(?:the\s+)?(?:figure|diagram|picture|image|graph|chart|screenshot)"
    r"(?:\s+(?:above|below|to\s+the\s+right|to\s+the\s+left|attached|provided))?\b|"
    r"\b(?:figure|diagram|picture|image|graph|chart|screenshot)\s+"
    r"(?:above|below|to\s+the\s+right|to\s+the\s+left|attached|provided)\b",
    re.IGNORECASE,
)

_QUARANTINE_REASONS = {
    "empty_statement",
    "invalid_statement_type",
    "example_only_statement",
    "missing_task_definition",
    "contradictory_numeric_bounds",
    "unresolved_image_placeholder",
    "non_unique_output",
    "numeric_tolerance",
    "case_insensitive_checker_required",
    "custom_checker_required",
}


def _issue(statement: str, reason: str, start: int, end: int) -> dict[str, object]:
    return {
        "reason": reason,
        "evidence": statement[start:end],
        "start": start,
        "end": end,
    }


def _section_name(match: re.Match[str]) -> str:
    return re.sub(r"\s+\d+$", "", match.group("name").lower())


def _analysis_text(statement: str) -> str:
    """Drop examples for task-definition analysis, never for evidence spans."""
    for match in _SECTION.finditer(statement):
        if _section_name(match) in {"example", "examples", "sample", "sample input", "sample output", "sample test", "sample test cases"}:
            return statement[:match.start()]
    return statement


def _content_after_header(text: str, header: re.Match[str], headers: list[re.Match[str]]) -> tuple[str, int, int]:
    next_start = len(text)
    for candidate in headers:
        if candidate.start() > header.start():
            next_start = candidate.start()
            break
    return text[header.end():next_start], header.end(), next_start


def _task_definition_issue(statement: str) -> dict[str, object] | None:
    text = _analysis_text(statement)
    headers = list(_SECTION.finditer(text))
    if not headers:
        return None

    first_spec_header = next(
        (match for match in headers if _section_name(match) in {"input", "output", "constraints"}),
        None,
    )
    if first_spec_header is None or text[:first_spec_header.start()].strip():
        return None

    input_header = next((match for match in headers if _section_name(match) == "input"), None)
    output_header = next((match for match in headers if _section_name(match) == "output"), None)
    if output_header is None:
        return None

    input_text = ""
    if input_header is not None:
        input_text, _, _ = _content_after_header(text, input_header, headers)
    output_text, output_start, _ = _content_after_header(text, output_header, headers)

    if _TASK_RELATION.search(input_text) or _TASK_RELATION.search(output_text) or _ARITHMETIC_OUTPUT.search(output_text):
        return None

    output_format = _CHOICE_ONLY_OUTPUT.fullmatch(output_text) or _TYPE_ONLY_OUTPUT.fullmatch(output_text)
    input_is_schema_only = bool(_BASIC_INPUT_SCHEMA.fullmatch(input_text))
    reason = "missing_task_definition" if output_format and input_is_schema_only else "task_definition_unclear"
    if output_format:
        claim_start, claim_end = output_format.span("claim")
        absolute_start = output_start + claim_start
        absolute_end = output_start + claim_end
    else:
        evidence = output_text.strip() or input_text.strip()
        if not evidence:
            return None
        local_start = (output_text if output_text.strip() else input_text).find(evidence)
        absolute_start = (output_start if output_text.strip() else input_header.end()) + local_start
        absolute_end = absolute_start + len(evidence)
    return _issue(statement, reason, absolute_start, absolute_end)


def _example_only_assessment(statement: str, match: re.Match[str]) -> dict[str, object]:
    tail_start = match.end()
    task_text = _TASK_RELATION.search(statement[tail_start:])
    if task_text:
        start = tail_start + task_text.start()
        end = tail_start + task_text.end()
        issue = _issue(statement, "task_text_after_example_heading", start, end)
        return {"status": "review_needed", "issues": [issue]}
    issue = _issue(statement, "example_only_statement", match.start("name"), match.end("name"))
    return {"status": "quarantine", "issues": [issue]}


def _bound_is_impossible(match: re.Match[str]) -> bool:
    left = int(re.sub(r"[, \t\xa0{}]", "", match.group("left")))
    right = int(re.sub(r"[, \t\xa0{}]", "", match.group("right")))
    operators = {
        "<": "lt", "<=": "le", "≤": "le", r"\le": "le", r"\leq": "le",
        ">": "gt", ">=": "ge", "≥": "ge", r"\ge": "ge", r"\geq": "ge",
    }
    first = operators[match.group("op1")]
    second = operators[match.group("op2")]
    lower_candidates: list[tuple[int, bool]] = []
    upper_candidates: list[tuple[int, bool]] = []

    if first in {"lt", "le"}:
        lower_candidates.append((left, first == "le"))
    else:
        upper_candidates.append((left, first == "ge"))
    if second in {"lt", "le"}:
        upper_candidates.append((right, second == "le"))
    else:
        lower_candidates.append((right, second == "ge"))

    if not lower_candidates or not upper_candidates:
        return False
    lower = max(value for value, _ in lower_candidates)
    upper = min(value for value, _ in upper_candidates)
    lower_closed = all(closed for value, closed in lower_candidates if value == lower)
    upper_closed = all(closed for value, closed in upper_candidates if value == upper)
    return lower > upper or (lower == upper and not (lower_closed and upper_closed))


def _bound_issues(statement: str) -> list[dict[str, object]]:
    text = _analysis_text(statement)
    headers = list(_SECTION.finditer(text))
    explicit_input_spans: list[tuple[int, int]] = []
    for header in headers:
        if _section_name(header) in {"input", "constraints"}:
            _, start, end = _content_after_header(text, header, headers)
            explicit_input_spans.append((start, end))

    findings: list[dict[str, object]] = []
    for match in _CHAINED_BOUNDS.finditer(text):
        if not _bound_is_impossible(match):
            continue
        explicit_constraint = any(
            start <= match.start() and match.end() <= end
            for start, end in explicit_input_spans
        )
        reason = "contradictory_numeric_bounds" if explicit_constraint else "contradictory_numeric_bounds_contextual"
        findings.append(_issue(statement, reason, match.start(), match.end()))
    return findings


def _unbalanced_math_issue(statement: str) -> dict[str, object] | None:
    dollars: list[int] = []
    for index, char in enumerate(statement):
        if char != "$":
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and statement[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            dollars.append(index)
    if len(dollars) % 2:
        index = dollars[-1]
        return _issue(statement, "unbalanced_math_delimiter", index, index + 1)

    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        tokens = list(re.finditer(r"(?<!\\)\\[()[\]]", statement))
        stack: list[int] = []
        unmatched_close: int | None = None
        for token in tokens:
            if token.group() == opening:
                stack.append(token.start())
            elif token.group() == closing:
                if stack:
                    stack.pop()
                elif unmatched_close is None:
                    unmatched_close = token.start()
        if unmatched_close is not None:
            return _issue(statement, "unbalanced_math_delimiter", unmatched_close, unmatched_close + 2)
        if stack:
            index = stack[-1]
            return _issue(statement, "unbalanced_math_delimiter", index, index + 2)
    return None


def assess_statement(statement: str) -> dict[str, object]:
    """Return a conservative triage status and exact statement evidence.

    Only clear defects are quarantined. ``review_needed`` records cues whose
    effect depends on source context. Unrecognized or difficult task grammars
    remain ``no_confirmed_issue`` unless a concrete pattern matches.
    """
    if not isinstance(statement, str):
        issue = {"reason": "invalid_statement_type", "evidence": "", "start": 0, "end": 0}
        return {"status": "quarantine", "issues": [issue]}
    if not statement.strip():
        issue = {"reason": "empty_statement", "evidence": "", "start": 0, "end": 0}
        return {"status": "quarantine", "issues": [issue]}

    headers = list(_SECTION.finditer(statement))
    first_nonspace = len(statement) - len(statement.lstrip())
    if headers and headers[0].start() == first_nonspace and _section_name(headers[0]) in {
        "example", "examples", "sample", "sample input", "sample output", "sample test", "sample test cases"
    }:
        return _example_only_assessment(statement, headers[0])

    issues: list[dict[str, object]] = []
    task_issue = _task_definition_issue(statement)
    if task_issue is not None:
        issues.append(task_issue)

    issues.extend(_bound_issues(statement))

    for finding in statement_issues(statement):
        start = int(finding["match_start"])
        end = int(finding["match_end"])
        issues.append(_issue(statement, str(finding["reason"]), start, end))

    visual = _VISUAL_REFERENCE.search(statement)
    if visual:
        issues.append(_issue(statement, "external_visual_reference", visual.start(), visual.end()))

    math_issue = _unbalanced_math_issue(statement)
    if math_issue is not None:
        issues.append(math_issue)

    unique: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for issue in issues:
        key = (issue["reason"], issue["start"], issue["end"])
        if key not in seen:
            seen.add(key)
            unique.append(issue)

    if any(issue["reason"] in _QUARANTINE_REASONS for issue in unique):
        status = "quarantine"
    elif unique:
        status = "review_needed"
    else:
        status = "no_confirmed_issue"
    return {"status": status, "issues": unique}
