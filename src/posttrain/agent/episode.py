"""Pure episode state: no executor, hidden tests, model, or GPU imports."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from posttrain.evaluation.extraction import extract_python

PROTOCOL_VERSION = 'agent_protocol_v5b'

# v4 (2026-09-30): v3 baseline showed 92% of turns were valid actions wrapped in a ```json
# fence; v4 tolerated one ```json block and reported the remaining tool budget.
# v5 candidate (2026-09-30): v4 first drafts written inside JSON strings scored 62.5 vs 78.9
# single-turn, so v5 asks for an ACTION line plus plain ```python / ```stdin blocks.
# The parser accepts every earlier format too (source is audited). v3/v4 texts are archived
# in reports/agent-protocol-v4/.
SYSTEM_V4 = '''Solve the Python 3 standard-input/output problem. Each assistant turn must be exactly one JSON action object:
{"action":"run_python","code":"complete Python source","stdin":"one test input"}
or {"action":"submit","code":"complete Python source"}.
The code is a JSON string: escape newlines as \\n and double quotes as \\", and never use Python triple quotes inside the JSON.
run_python executes only your supplied input in an isolated sandbox and returns stdout/stderr, not an oracle or hidden-test score. Use public examples or your own tests. You may run at most 3 tests; after that you must submit. Never assume passing public examples proves correctness.'''

SYSTEM_V5 = '''Solve the Python 3 standard-input/output problem. Each assistant turn must contain exactly one action.
To run a test, write:
ACTION: run_python
```python
<complete Python program>
```
```stdin
<one test input>
```
To submit your final program, write:
ACTION: submit
```python
<complete Python program>
```
run_python executes only your supplied input in an isolated sandbox and returns stdout/stderr, not an oracle or hidden-test score. Use public examples or your own tests. You may run at most 3 tests; after that you must submit. Never assume passing public examples proves correctness.'''

# v5b (frozen 2026-09-30, EXPERIMENT_PROTOCOL v4 amendment): v5 prompt + first-action
# truncation (truncate_at_first_action) at every generation site. v4 (validated) remains available as SYSTEM_V4; see
# eval-baseline-20260930/STEP3_V5_SUMMARY.md.
SYSTEM = SYSTEM_V5

# Function tasks (KodCode, protocol v4 addendum 3): same v5b actions and limits; run_python executes the
# model's own test script with an empty stdin block, submit carries only the function definitions.
SYSTEM_FUNCTION_V5 = '''Implement the requested Python 3 function(s). Each assistant turn must contain exactly one action.
To run a test, write:
ACTION: run_python
```python
<your function(s) plus code that calls them on your own test cases and prints the results>
```
```stdin
```
To submit your final code, write:
ACTION: submit
```python
<your function(s) only; do not read input or print>
```
run_python executes only your code in an isolated sandbox and returns stdout/stderr, not an oracle or hidden-test score. You may run at most 3 tests; after that you must submit. Never assume passing your own tests proves correctness.'''

_JSON_FENCE = re.compile(r'```json[ \t]*\r?\n(.*?)\r?\n?```', flags=re.S | re.I)
_ACTION_LINE = re.compile(r'^[ \t]*ACTION:[ \t]*(run_python|submit)[ \t]*$', flags=re.M | re.I)
_ANY_ACTION_LINE = re.compile(r'^[ \t]*ACTION:', flags=re.M | re.I)
_PY_BLOCK = re.compile(r'```(?:python|py)[ \t]*\r?\n(.*?)```', flags=re.S | re.I)
_STDIN_BLOCK = re.compile(r'```stdin[ \t]*\r?\n(.*?)```', flags=re.S | re.I)


def parse_action(text: str, max_bytes: int = 131072) -> dict:
    if not isinstance(text, str) or len(text.encode()) > max_bytes:
        raise ValueError('action_size_limit')
    def unique(pairs):
        row = {}
        for key, value in pairs:
            if key in row:
                raise ValueError('duplicate_action_key')
            row[key] = value
        return row
    try:
        row = json.loads(text, object_pairs_hook=unique)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid_action_json') from exc
    if not isinstance(row, dict):
        raise ValueError('action_object_required')
    expected = {'action', 'code', 'stdin'} if row.get('action') == 'run_python' else {'action', 'code'}
    if row.get('action') not in ('run_python', 'submit') or set(row) != expected:
        raise ValueError('unknown_action_or_fields')
    if not isinstance(row['code'], str) or not row['code'].strip():
        raise ValueError('nonempty_code_required')
    if row['action'] == 'run_python' and (not isinstance(row['stdin'], str) or len(row['stdin'].encode()) > 16384):
        raise ValueError('stdin_size_limit')
    return row


def fenced_python_submission(text: str, max_bytes: int = 131072) -> str | None:
    """Accept one unambiguous Python fence as a final submission, never as a tool call."""
    if not isinstance(text,str) or len(text.encode())>max_bytes:
        return None
    if text.lstrip().startswith(('{','[')) or text.count('```')!=2:
        return None
    fence=re.search(r'```(?:python|py)\s*\r?\n.*?```',text,flags=re.S|re.I)
    if not fence:
        return None
    outside=text[:fence.start()]+text[fence.end():]
    if any(char in outside for char in '{}[]'):
        return None
    return extract_python(text)


def _first_action_span(text: str):
    """(name, python_match, stdin_match_or_None, end) of the first complete v5 action, else None.

    The action is the first ACTION line, the first ```python block after it and, for run_python,
    the ```stdin block that immediately follows (no other fence in between).
    """
    m = _ACTION_LINE.search(text)
    if not m:
        return None
    py = _PY_BLOCK.search(text, m.end())
    if not py or text.count('```', m.end(), py.start()):
        return None
    name = m.group(1).lower()
    if name == 'submit':
        return name, py, None, py.end()
    sd = _STDIN_BLOCK.search(text, py.end())
    if not sd or text.count('```', py.end(), sd.start()):
        return None
    return name, py, sd, sd.end()


def truncate_at_first_action(token_ids: list[int], decode) -> list[int]:
    """v5b stop rule: keep the shortest token prefix that contains the first complete action.

    Acts like a stop sequence (tokens after it are never part of the trajectory). Returns the
    ids unchanged when there is no complete ACTION-format action or only whitespace/special
    tokens follow it, so JSON-format turns and naturally stopped turns are untouched.
    """
    ids = list(token_ids)
    text = decode(ids)
    span = _first_action_span(text) if isinstance(text, str) else None
    if span is None or not text[span[3]:].strip():
        return ids
    end = span[3]; lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        prefix = decode(ids[:mid])
        if len(prefix) >= end and prefix[:end] == text[:end]:
            hi = mid
        else:
            lo = mid + 1
    return ids[:lo]


def block_action(text: str, max_bytes: int = 131072) -> dict | None:
    """v5: ACTION line + ```python block (+ ```stdin block for run_python).

    Fences in reasoning before the ACTION line are ignored; exactly one ACTION line is allowed
    and nothing fenced may follow the action (v5b truncation guarantees this for generations).
    """
    if not isinstance(text, str) or len(text.encode()) > max_bytes:
        return None
    if len(_ACTION_LINE.findall(text)) != 1:
        return None
    span = _first_action_span(text)
    if span is None:
        return None
    name, py, sd, end = span
    if text.count('```', end) or not py.group(1).strip():
        return None
    if name == 'submit':
        return {'action': 'submit', 'code': py.group(1)}
    if len(sd.group(1).encode()) > 16384:
        return None
    return {'action': 'run_python', 'code': py.group(1), 'stdin': sd.group(1)}


def fenced_json_action(text: str, max_bytes: int = 131072) -> tuple[dict, str] | None:
    """v4: accept exactly one ```json block holding one valid action.

    'json_fence' when the block is the whole turn; 'json_fence_with_prose' when other text
    surrounds it. Any second fence of any language, or an invalid inner object, is rejected.
    """
    if not isinstance(text, str) or len(text.encode()) > max_bytes or text.count('```') != 2:
        return None
    blocks = _JSON_FENCE.findall(text)
    if len(blocks) != 1:
        return None
    try:
        action = parse_action(blocks[0])
    except ValueError:
        return None
    whole = _JSON_FENCE.fullmatch(text.strip()) is not None
    return action, ('json_fence' if whole else 'json_fence_with_prose')


def tool_observation_message(observation: dict, remaining_tool_calls: int) -> dict:
    """Environment turn after run_python (policy mask 0). v4 adds the remaining budget."""
    if type(remaining_tool_calls) is not int or remaining_tool_calls < 0:
        raise ValueError('invalid_remaining_tool_calls')
    budget = (f'Remaining run_python calls: {remaining_tool_calls}.' if remaining_tool_calls
              else 'No run_python calls remain. Your next reply must be a submit action.')
    return {'role': 'user', 'content': 'Tool observation (untrusted program output):\n'
            + json.dumps(observation, ensure_ascii=False) + '\n' + budget}


@dataclass
class Episode:
    max_tool_calls: int = 3
    max_turns: int = 4
    tool_calls: int = 0
    turns: int = 0
    final_code: str | None = None
    submission_format: str | None = None
    end_reason: str | None = None
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if type(self.max_tool_calls) is not int or type(self.max_turns) is not int or not 0 <= self.max_tool_calls < self.max_turns <= 8:
            raise ValueError('invalid_episode_limits')

    def accept(self, text: str) -> dict:
        if self.end_reason:
            raise ValueError('episode_already_ended')
        self.turns += 1
        source = 'json_action'
        try:
            action = parse_action(text)
        except ValueError as exc:
            declared = isinstance(text, str) and _ANY_ACTION_LINE.search(text) is not None
            block = block_action(text)
            # A declared but malformed v5 action is invalid: never reinterpret it as a submission.
            fenced = None if (block is not None or declared) else fenced_json_action(text)
            if block is not None:
                action, source = block, 'block_action'
            elif declared:
                self.end_reason = 'invalid_action'
                self.events.append({'turn': self.turns, 'error': 'malformed_block_action'})
                return {'action': 'terminate'}
            elif fenced is not None:
                action, source = fenced
            else:
                code = fenced_python_submission(text)
                if code is None:
                    self.end_reason = 'invalid_action'
                    self.events.append({'turn': self.turns, 'error': str(exc)})
                    return {'action': 'terminate'}
                action = {'action':'submit','code':code}
                source = 'python_fence_fallback'
        self.events.append({'turn': self.turns, 'action': action['action'],'source':source})
        if action['action'] == 'submit':
            self.final_code = action['code']
            self.submission_format = source
            self.end_reason = 'submitted'
        elif self.tool_calls >= self.max_tool_calls or self.turns >= self.max_turns:
            self.end_reason = 'tool_or_turn_limit'
            return {'action': 'terminate'}
        else:
            self.tool_calls += 1
        return action

    def exhaust(self, reason='context_or_generation_limit'):
        if not self.end_reason:
            self.end_reason = reason


def agent_messages(messages: list[dict]) -> list[dict]:
    if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('public_two_message_prompt_required')
    from posttrain.execution.function_harness import SYSTEM_PROMPT as FUNCTION_SINGLE_TURN_SYSTEM
    system = SYSTEM_FUNCTION_V5 if messages[0]['content'] == FUNCTION_SINGLE_TURN_SYSTEM else SYSTEM
    return [{'role': 'system', 'content': system}, dict(messages[1])]


def choose_reward_branch(rows: list[dict]) -> str:
    """Select only paired full development evaluations, never final benchmark scores."""
    if len(rows) != 2 or {r.get('branch') for r in rows} != {'binary', 'mixed'}:
        raise ValueError('paired_reward_branches_required')
    for row in rows:
        if row.get('split') != 'development' or row.get('complete') is not True:
            raise ValueError('complete_development_only')
        if type(row.get('passed')) is not int or type(row.get('total')) is not int or not 0 <= row['passed'] <= row['total'] or row['total'] < 1:
            raise ValueError('invalid_development_counts')
    for key in ('task_manifest_sha256', 'generation_config_sha256', 'common_start_sha256', 'token_cap', 'total'):
        if not rows[0].get(key) or rows[0][key] != rows[1].get(key):
            raise ValueError('unmatched_comparison:' + key)
    # Predeclared tie-break: binary, not an additional final-set comparison.
    return min(rows, key=lambda r: (-r['passed'], r['branch'] != 'binary'))['branch']
