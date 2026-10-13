from __future__ import annotations

from collections import deque

from posttrain.execution.contracts import CaseResult


class ReplayBackend:
    """Contract/fault tests ONLY. It never parses or runs the supplied program."""
    version='scripted-replay-v1'
    evidence_level='mock_contract'

    def __init__(self, outcomes: list[CaseResult]):
        self.outcomes=deque(outcomes)
        self.calls=[]

    def run(self, code, stdin, limits, request_id, timeout_s):
        self.calls.append({'request_id':request_id,'stdin':stdin,'timeout_s':timeout_s})
        if not self.outcomes:
            raise AssertionError('replay_script_exhausted')
        return self.outcomes.popleft()
