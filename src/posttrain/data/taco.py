from __future__ import annotations

import json
import re

from posttrain.data.schema import DataError, Problem, TestCase
from posttrain.data.eligibility import statement_issues


def parse_json(value, label):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as e:
        raise DataError(f"malformed_{label}") from e


def from_taco(row: dict, revision: str, row_index: int, min_tests: int = 5) -> Problem:
    if row.get("picture_num") not in (None, "", "0", 0, "[]"):
        raise DataError("requires_image")
    statement = row.get("question", "")
    if not isinstance(statement, str) or not statement.strip():
        raise DataError("missing_statement")
    issues=statement_issues(statement)
    if issues:raise DataError(issues[0]['reason'])
    if re.search(r"\b(?:this is an interactive problem|interactive task)\b", statement, re.I):
        raise DataError("interactive_problem")
    io = parse_json(row.get("input_output"), "tests")
    if not isinstance(io, dict) or io.get("fn_name"):
        raise DataError("unsupported_interface")
    inputs, outputs = io.get("inputs"), io.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise DataError("malformed_tests")
    if any(not isinstance(x, str) for x in inputs + outputs):
        raise DataError("nontext_stdio_test")
    solutions = parse_json(row.get("solutions"), "solutions")
    if not isinstance(solutions, list):
        raise DataError("missing_references")
    # Do not eval source tags or any other Python-valued strings.
    refs = tuple(x for x in solutions if isinstance(x, str) and x.strip())
    limit = str(row.get("time_limit") or "2")
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", limit)
    seconds = float(match.group()) if match else 2.0
    if "ms" in limit.lower() or "millisecond" in limit.lower():
        seconds /= 1000
    source_id = str(row.get("url") or f"{row.get('source', 'TACO')}:{row.get('name') or row_index}")
    problem = Problem(problem_id=f"taco-train-{row_index:06d}", source="BAAI/TACO", revision=revision,
                      source_id=source_id, statement=statement, tests=tuple(TestCase(a,b) for a,b in zip(inputs,outputs)),
                      references=refs, difficulty=str(row.get("difficulty") or "unknown"),
                      time_limit_s=min(max(seconds, 0.1), 30), metadata={"raw_row_index":row_index, "raw_time_limit":limit})
    if len(problem.tests) < min_tests:
        raise DataError("too_few_unique_tests")
    return problem
