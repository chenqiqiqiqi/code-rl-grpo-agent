from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from posttrain.common import digest


class Status(str, Enum):
    OK = "ok"
    WRONG = "wrong_answer"
    SYNTAX = "syntax_error"
    RUNTIME = "runtime_error"
    TIMEOUT = "program_timeout"
    MEMORY = "memory_limit"
    OUTPUT = "output_limit"
    INFRA = "infrastructure_error"
    DATA = "data_error"


@dataclass(frozen=True)
class Limits:
    time_s: float = 2.0
    memory_mb: int = 256
    output_bytes: int = 65536
    pids: int = 16
    cpu_cores: int = 1

    def __post_init__(self):
        if not 0 < self.time_s <= 30 or any(type(v) is not int or v <= 0 for v in (self.memory_mb,self.output_bytes,self.pids,self.cpu_cores)):
            raise ValueError("invalid_resource_limits")


@dataclass(frozen=True)
class CaseResult:
    status: Status
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    elapsed_s: float = 0.0
    queue_s: float | None = None
    physical_executions: int | None = 1
    truncated: bool = False
    detail: str = ""


@dataclass(frozen=True)
class JudgeResult:
    request_id: str
    status: Status
    passed: int
    total: int
    cases: tuple[CaseResult, ...]
    elapsed_s: float
    queue_s: float | None
    physical_executions: int | None
    retries: int
    cache_hit: bool
    executor_version: str
    evidence_level: str
    code_hash: str
    tests_hash: str
    scoring_mode: str = 'full'

    @property
    def valid(self) -> bool:
        return (self.status not in (Status.INFRA, Status.DATA) and self.total > 0
                and len(self.cases) == self.total and 0 <= self.passed <= self.total
                and self.passed == sum(c.status==Status.OK for c in self.cases)
                and (self.status==Status.OK)==(self.passed==self.total)
                and all(c.status not in (Status.INFRA,Status.DATA) for c in self.cases))

    @property
    def binary_valid(self) -> bool:
        """A confirmed failure decides binary reward, not the untested pass rate."""
        if self.valid:
            return True
        return (self.scoring_mode == 'binary_fail_fast' and self.total > 0
                and 0 < len(self.cases) < self.total
                and self.status not in (Status.OK, Status.INFRA, Status.DATA)
                and self.passed == sum(c.status == Status.OK for c in self.cases)
                and self.passed < len(self.cases)
                and all(c.status not in (Status.INFRA, Status.DATA) for c in self.cases))

    def to_dict(self):
        result = asdict(self)
        result['valid'] = self.valid
        result['binary_valid'] = self.binary_valid
        result['skipped_case_judgments'] = self.total-len(self.cases)
        result['planned_case_judgments'] = self.total
        result['completed_case_judgments'] = len(self.cases)
        return result


def cache_key(code: str, tests_hash: str, limits: Limits, executor_version: str) -> str:
    return digest({"code":code,"tests":tests_hash,"limits":asdict(limits),"executor":executor_version,
                   "comparator":"exact_whitespace_tokens_v1"})
