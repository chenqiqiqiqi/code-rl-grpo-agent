from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from posttrain.common import digest


class DataError(ValueError):
    pass


def normalize_output(text: str) -> str:
    """Frozen stdio contract: exact whitespace-delimited tokens, no numeric tolerance."""
    return " ".join(text.split())


@dataclass(frozen=True)
class TestCase:
    stdin: str
    stdout: str

    def __post_init__(self):
        if not isinstance(self.stdin, str) or not isinstance(self.stdout, str):
            raise DataError("stdio tests require text input and output")


def clean_tests(tests: tuple[TestCase, ...]) -> tuple[TestCase, ...]:
    if not tests:
        raise DataError("empty_tests")
    seen: dict[str, str] = {}
    result = []
    for test in tests:
        # Preserve input whitespace: it can be semantically meaningful.
        key = test.stdin.replace("\r\n", "\n")
        expected = normalize_output(test.stdout)
        if key in seen and seen[key] != expected:
            raise DataError("conflicting_tests")
        if key not in seen:
            result.append(test)
        seen[key] = expected
    return tuple(result)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    source: str
    revision: str
    source_id: str
    statement: str
    tests: tuple[TestCase, ...]
    references: tuple[str, ...]
    difficulty: str = "unknown"
    interface: Literal["stdio"] = "stdio"
    public_examples: tuple[TestCase, ...] = ()
    time_limit_s: float = 2.0
    cluster_id: str = ""
    validation_status: str = "unverified"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("problem_id", "source", "revision", "source_id", "statement"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise DataError(f"missing_{name}")
        if self.interface != "stdio":
            raise DataError("unsupported_interface")
        if not math.isfinite(self.time_limit_s) or not 0 < self.time_limit_s <= 30:
            raise DataError("invalid_time_limit")
        if not self.references or any(not isinstance(x, str) or not x.strip() for x in self.references):
            raise DataError("missing_references")
        object.__setattr__(self, "tests", clean_tests(self.tests))

    @property
    def tests_hash(self) -> str:
        return digest([asdict(t) for t in self.tests])

    def prompt(self) -> list[dict[str, str]]:
        # Whitelist serialization: never dump the Problem/dataclass into a prompt.
        examples = "".join(f"\nSample input:\n{x.stdin}\nSample output:\n{x.stdout}" for x in self.public_examples)
        return [{"role": "system", "content": "Write a complete Python 3 program. Read standard input and write standard output. Return code only."},
                {"role": "user", "content": self.statement + examples}]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "Problem":
        data = dict(row)
        for key in ("tests", "public_examples"):
            data[key] = tuple(TestCase(**t) for t in data.get(key, []))
        data["references"] = tuple(data.get("references", []))
        return cls(**data)
