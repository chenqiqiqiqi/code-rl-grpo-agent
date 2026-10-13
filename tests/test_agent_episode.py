import json
import pytest
from posttrain.agent.episode import Episode, parse_action, agent_messages, choose_reward_branch
from posttrain.agent.tool_service import checked_request, ToolClient


def test_tool_then_submit_and_limits():
    e = Episode()
    for _ in range(3):
        assert e.accept(json.dumps({'action':'run_python','code':'print(1)','stdin':''}))['action']=='run_python'
    assert not e.end_reason
    e.accept(json.dumps({'action':'submit','code':'print(1)'}))
    assert e.final_code=='print(1)' and e.end_reason=='submitted'
    assert e.submission_format=='json_action'
    with pytest.raises(ValueError): e.accept('{}')
    e=Episode(max_tool_calls=0,max_turns=1)
    assert e.accept(json.dumps({'action':'run_python','code':'print(1)','stdin':''}))['action']=='terminate'
    assert e.tool_calls==0 and e.final_code is None


@pytest.mark.parametrize('text',['[]','{"action":"submit","action":"run_python","code":"x"}',
    '{"action":"submit","code":"x","hidden_tests":[]}',
    '{"action":"shell","code":"x"}', '{"action":"submit","code":""}'])
def test_invalid_actions_terminate_without_execution(text):
    e=Episode()
    assert e.accept(text)=={'action':'terminate'}
    assert e.end_reason=='invalid_action' and e.final_code is None


def test_single_python_fence_is_audited_final_submission():
    e=Episode()
    action=e.accept('Here is my solution:\n```python\nprint(42)\n```\n')
    assert action=={'action':'submit','code':'print(42)\n'}
    assert e.final_code=='print(42)\n' and e.end_reason=='submitted'
    assert e.submission_format=='python_fence_fallback'
    assert e.events==[{'turn':1,'action':'submit','source':'python_fence_fallback'}]


@pytest.mark.parametrize('text',[
    '```python\nprint(1)\n```\n```python\nprint(2)\n```',
    '{"action":"submit","code":"print(1)"}\n```python\nprint(2)\n```',
    'Explanation:\n{bad JSON}\n```python\nprint(2)\n```',
    '```python\ndef bad(\n```',
    'print(42)',
])
def test_ambiguous_or_unfenced_fallback_is_still_invalid(text):
    e=Episode()
    assert e.accept(text)=={'action':'terminate'}
    assert e.end_reason=='invalid_action' and e.final_code is None


RUN = {'action': 'run_python', 'code': 'print(1)', 'stdin': '3\n'}


def test_v4_single_json_fence_is_accepted_and_audited():
    e = Episode()
    assert e.accept('```json\n' + json.dumps(RUN, indent=2) + '\n```') == RUN
    assert e.events == [{'turn': 1, 'action': 'run_python', 'source': 'json_fence'}]
    e.accept('Output matches.\n```json\n{"action":"submit","code":"print(1)"}\n```\n')
    assert e.end_reason == 'submitted' and e.submission_format == 'json_fence_with_prose'


@pytest.mark.parametrize('text', [
    '```json\n{"action":"submit","code":"a"}\n```\n```json\n{"action":"submit","code":"b"}\n```',
    '```json\n{"action":"submit"}\n```',
    '```json\n{"action":"submit","code":"""x"""}\n```',
    '```json\n{"action":"submit","code":"a"}\n```\n```python\nprint(2)\n```',
])
def test_v4_ambiguous_or_invalid_json_fence_still_terminates(text):
    e = Episode()
    assert e.accept(text) == {'action': 'terminate'}
    assert e.end_reason == 'invalid_action' and e.final_code is None


def test_v5_block_actions_run_then_submit():
    e = Episode()
    run = e.accept('Let me test.\nACTION: run_python\n```python\nprint(input())\n```\n```stdin\n3\n```\n')
    assert run == {'action': 'run_python', 'code': 'print(input())\n', 'stdin': '3\n'}
    sub = e.accept('ACTION: submit\n```python\nprint(input())\n```')
    assert sub == {'action': 'submit', 'code': 'print(input())\n'}
    assert e.end_reason == 'submitted' and e.submission_format == 'block_action'
    assert [x['source'] for x in e.events] == ['block_action', 'block_action']


@pytest.mark.parametrize('text', [
    'ACTION: run_python\n```python\nprint(1)\n```',                                    # missing stdin
    'ACTION: run_python\n```python\nprint(1)\n```\n```stdin\n1\n```\n```stdin\n2\n```',  # two inputs
    'ACTION: submit\n```python\nprint(1)\n```\n```python\nprint(2)\n```',              # two programs
    'ACTION: submit\nACTION: run_python\n```python\nprint(1)\n```\n```stdin\n1\n```',  # two actions
    'ACTION: shell\n```python\nprint(1)\n```',                                          # unknown action
    'ACTION: submit\n```python\n\n```',                                                  # empty code
])
def test_v5_ambiguous_block_actions_terminate(text):
    e = Episode()
    assert e.accept(text) == {'action': 'terminate'}
    assert e.end_reason == 'invalid_action' and e.final_code is None


def _chars(text):  # char-level fake tokenizer: one id per character
    return [ord(c) for c in text], (lambda ids: ''.join(map(chr, ids)))


def test_v5b_truncates_after_first_complete_action():
    from posttrain.agent.episode import truncate_at_first_action
    first = 'I will test.\nACTION: run_python\n```python\nprint(input())\n```\n```stdin\n3\n```'
    ids, dec = _chars(first + '\nOutput: 3\nACTION: submit\n```python\nprint(input())\n```\n')
    kept = truncate_at_first_action(ids, dec)
    assert dec(kept) == first
    e = Episode()
    assert e.accept(dec(kept)) == {'action': 'run_python', 'code': 'print(input())\n', 'stdin': '3\n'}
    ids2, _ = _chars(first + '\n\n')            # only whitespace after: unchanged
    assert truncate_at_first_action(ids2, dec) == ids2
    ids3, _ = _chars('{"action":"submit","code":"print(1)"} extra')  # JSON turns: unchanged
    assert truncate_at_first_action(ids3, dec) == ids3
    assert truncate_at_first_action(kept, dec) == kept   # idempotent (replay safety)


def test_v5b_multiple_stdin_blocks_keep_only_first():
    from posttrain.agent.episode import truncate_at_first_action
    text = 'ACTION: run_python\n```python\nprint(1)\n```\n```stdin\n1\n```\n```stdin\n2\n```'
    ids, dec = _chars(text)
    e = Episode()
    assert e.accept(dec(truncate_at_first_action(ids, dec)))['stdin'] == '1\n'


def test_v5_reasoning_fence_before_action_is_ignored():
    e = Episode()
    text = 'Draft:\n```python\nprint(0)\n```\nFixed.\nACTION: submit\n```python\nprint(1)\n```'
    assert e.accept(text) == {'action': 'submit', 'code': 'print(1)\n'}


def test_v4_tool_observation_reports_remaining_budget():
    from posttrain.agent.episode import tool_observation_message, SYSTEM_V4 as SYSTEM, PROTOCOL_VERSION
    obs = {'status': 'ok', 'stdout': '1\n', 'stderr': '', 'exit_code': 0, 'truncated': False}
    msg = tool_observation_message(obs, 2)
    assert msg['role'] == 'user' and msg['content'].startswith('Tool observation (untrusted program output):\n')
    assert json.dumps(obs) in msg['content'] and msg['content'].endswith('Remaining run_python calls: 2.')
    assert tool_observation_message(obs, 0)['content'].endswith('Your next reply must be a submit action.')
    with pytest.raises(ValueError): tool_observation_message(obs, -1)
    assert PROTOCOL_VERSION.startswith('agent_protocol_v') and 'triple quotes' in SYSTEM


def test_prompt_whitelist_and_tool_request():
    messages=agent_messages([{'role':'system','content':'old'},{'role':'user','content':'public'}])
    assert messages[1]['content']=='public'
    with pytest.raises(ValueError):checked_request({'code':'x','stdin':'','request_id':'a'*32,'tests':[]})
    with pytest.raises(ValueError):ToolClient('http://remote.example:8081','version')


def test_branch_selection_dev_only_tie_and_matching_budget():
    rows=[dict(branch=b,split='development',complete=True,passed=10,total=20,
               task_manifest_sha256='a',generation_config_sha256='b',common_start_sha256='c',token_cap=100) for b in ('binary','mixed')]
    assert choose_reward_branch(rows)=='binary'
    rows[1]['passed']=11
    assert choose_reward_branch(rows)=='mixed'
    rows[1]['split']='final'
    with pytest.raises(ValueError):choose_reward_branch(rows)
    rows[1]['split']='development'; rows[1]['token_cap']=101
    with pytest.raises(ValueError):choose_reward_branch(rows)
