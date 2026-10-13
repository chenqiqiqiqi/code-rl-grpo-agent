from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Protocol

from posttrain.common import digest
from posttrain.data.schema import DataError, TestCase, clean_tests, normalize_output
from posttrain.execution.contracts import CaseResult, JudgeResult, Limits, Status, cache_key


class Backend(Protocol):
    version: str
    evidence_level: str
    def run(self, code: str, stdin: str, limits: Limits, request_id: str, timeout_s: float) -> CaseResult: ...


class Judge:
    def __init__(self, backend: Backend, max_retries: int = 2, deadline_s: float | None = None,
                 backoff_s: float = .1, clock=time.monotonic, sleep=time.sleep):
        if max_retries not in (0,1,2) or (deadline_s is not None and deadline_s <= 0) or backoff_s < 0:
            raise ValueError("invalid_retry_policy")
        self.backend, self.max_retries, self.deadline_s = backend,max_retries,deadline_s
        self.backoff_s,self.clock,self.sleep = backoff_s,clock,sleep
        self.cache: dict[str,JudgeResult] = {}

    def score(self, code: str, tests: tuple[TestCase,...], limits: Limits, request_id: str | None = None,
              *, scoring_mode: str = 'full') -> JudgeResult:
        if scoring_mode not in ('full', 'binary_fail_fast'):
            raise ValueError('invalid_scoring_mode')
        request_id = request_id or uuid.uuid4().hex
        start = self.clock()
        original_total = len(tests)
        try:
            tests = clean_tests(tests)
        except DataError as e:
            return JudgeResult(request_id,Status.DATA,0,original_total,(),0,0,0,0,False,
                               self.backend.version,self.backend.evidence_level,digest(code),digest(str(e)),scoring_mode)
        thash = digest([{"stdin":x.stdin,"stdout":x.stdout} for x in tests])
        key = cache_key(code,thash,limits,self.backend.version)+':'+scoring_mode
        if key in self.cache:
            # Cached outcomes retain their verdicts, but perform no new case work.
            # Copy nested records so the original execution evidence stays intact.
            return replace(self.cache[key],request_id=request_id,cache_hit=True,elapsed_s=self.clock()-start,
                           queue_s=0,physical_executions=0,retries=0,
                           cases=tuple(replace(c,physical_executions=0,elapsed_s=0,queue_s=0)
                                       for c in self.cache[key].cases))
        # Program timeouts must still receive a complete score. Admit the full
        # frozen test allowance before dispatch; do not discover a bad deadline
        # by converting ordinary slow/wrong programs into invalid groups.
        deadline = self.deadline_s if self.deadline_s is not None else (
            len(tests)*(limits.time_s+1)+60+self.max_retries*(limits.time_s+30))
        if deadline<len(tests)*limits.time_s:
            raise ValueError('deadline_cannot_cover_full_tests')
        cases, retries, physical, queue = [],0,0,0.0
        physical_known, queue_known = True,True
        def account(result):
            nonlocal physical_known,physical,queue_known,queue
            physical_known &= result.physical_executions is not None
            physical += result.physical_executions or 0
            queue_known &= result.queue_s is not None
            queue += result.queue_s or 0

        initial = None
        if scoring_mode == 'full' and callable(getattr(self.backend,'run_batch',None)):
            initial = self.backend.run_batch(code,[c.stdin for c in tests],limits,
                                             request_id,deadline-(self.clock()-start))
            if not isinstance(initial,list) or len(initial)!=len(tests) or not all(isinstance(r,CaseResult) for r in initial):
                initial = [CaseResult(Status.INFRA,physical_executions=None,detail='malformed_batch_response') for _ in tests]
            # Account work already performed even if an earlier case stays invalid.
            for result in initial:
                account(result)
        chunk, chunk_end = {}, 0
        for i,case in enumerate(tests):
            if initial is not None:
                result = initial[i]
            elif scoring_mode == 'binary_fail_fast' and callable(getattr(self.backend,'run_batch',None)):
                if i == chunk_end:
                    # One cheap first probe, then bounded chunks. Expected outputs
                    # stay in this controller; the existing isolated image is unchanged.
                    chunk_end = min(len(tests), i+(1 if i == 0 else 8))
                    remaining = deadline-(self.clock()-start)
                    batch = (self.backend.run_batch(code,[c.stdin for c in tests[i:chunk_end]],limits,
                        f'{request_id}:chunk:{i}',remaining) if remaining > 0 else None)
                    if (not isinstance(batch,list) or len(batch)!=chunk_end-i
                            or not all(isinstance(r,CaseResult) for r in batch)):
                        batch = [CaseResult(Status.INFRA,physical_executions=None,
                            detail='malformed_or_expired_batch') for _ in tests[i:chunk_end]]
                    chunk = dict(zip(range(i,chunk_end),batch))
                    for item in batch:account(item)
                result = chunk[i]
            else:
                remaining = deadline-(self.clock()-start)
                if remaining <= 0:
                    result = CaseResult(Status.INFRA,physical_executions=0,detail="total_deadline")
                else:
                    result = self.backend.run(code,case.stdin,limits,f"{request_id}:{i}:0",remaining)
                account(result)
            while result.status==Status.INFRA and retries<self.max_retries:
                delay = self.backoff_s * 2**retries
                if delay >= deadline-(self.clock()-start):
                    break
                self.sleep(delay)
                retries += 1
                result = self.backend.run(code,case.stdin,limits,f"{request_id}:{i}:retry{retries}",
                                           deadline-(self.clock()-start))
                account(result)
            if result.status == Status.OK and normalize_output(result.stdout) != normalize_output(case.stdout):
                result = replace(result,status=Status.WRONG)
            cases.append(result)
            # Consume all already-dispatched cases, including any infrastructure
            # failure later in the chunk, before accepting a binary verdict.
            boundary = not chunk or i+1 == chunk_end
            if (scoring_mode == 'binary_fail_fast' and boundary
                    and any(c.status != Status.OK for c in cases)):
                break
            if result.status in (Status.INFRA,Status.DATA) and initial is None and not chunk:
                break
        failures = [c.status for c in cases if c.status != Status.OK]
        status = next((x for x in failures if x in (Status.INFRA,Status.DATA)), failures[0] if failures else Status.OK)
        scored = JudgeResult(request_id,status,sum(c.status==Status.OK for c in cases),len(tests),tuple(cases),
                             self.clock()-start,queue if queue_known else None,physical if physical_known else None,
                             retries,False,self.backend.version,self.backend.evidence_level,digest(code),thash,scoring_mode)
        if scored.valid or scored.binary_valid:
            self.cache[key] = scored
        return scored
