"""Agent protocol v5b on function tasks (KodCode): system prompt choice and empty-stdin test actions."""
from posttrain.agent.episode import SYSTEM, SYSTEM_FUNCTION_V5, Episode, agent_messages, block_action, truncate_at_first_action
from posttrain.execution.function_harness import SYSTEM_PROMPT


def test_function_prompts_get_the_function_agent_system_and_stdio_prompts_are_unchanged():
    func = agent_messages([{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': 'Write add(a, b).'}])
    assert func[0]['content'] == SYSTEM_FUNCTION_V5 and func[1]['content'] == 'Write add(a, b).'
    stdio = agent_messages([{'role': 'system', 'content': 'Write a complete Python 3 program. Read standard input and '
                             'write standard output. Return code only.'}, {'role': 'user', 'content': 'p'}])
    assert stdio[0]['content'] == SYSTEM


def test_function_style_test_action_with_empty_stdin_and_submit_parse():
    run = 'Let me test.\nACTION: run_python\n```python\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```\n```stdin\n```\n'
    assert block_action(run) == {'action': 'run_python', 'code': 'def add(a, b):\n    return a + b\nprint(add(1, 2))\n', 'stdin': ''}
    episode = Episode()
    assert episode.accept(run)['action'] == 'run_python'
    sub = 'ACTION: submit\n```python\ndef add(a, b):\n    return a + b\n```'
    assert block_action(sub) == {'action': 'submit', 'code': 'def add(a, b):\n    return a + b\n'}


def test_first_action_truncation_cuts_an_invented_tool_result():
    text = ('ACTION: run_python\n```python\nprint(1)\n```\n```stdin\n```\nOutput: 1\nACTION: submit\n```python\nx=1\n```')
    ids = list(range(len(text)))
    kept = truncate_at_first_action(ids, lambda xs: text[:len(xs)])
    assert text[:len(kept)].rstrip().endswith('```stdin\n```')
