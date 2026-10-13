"""Function-task harness: run the real wrapped program in a local subprocess (no sandbox needed)."""
import subprocess
import sys

import pytest

from posttrain.execution.function_harness import hidden_programs, is_function_task, wrap

TESTS = '''from solution import add

def test_small():
    assert add(1, 2) == 3

def test_negative():
    assert add(-1, -1) == -2
'''
TOKEN = 'A1b2C3d4E5f6G7h8'


def run(code, stdin):
    return subprocess.run([sys.executable, '-c', code], input=stdin, capture_output=True, text=True, timeout=20).stdout


def outcomes(candidate):
    return [run(wrap(candidate), stdin) == expected for stdin, expected in hidden_programs(TESTS, TOKEN)]


def test_correct_candidate_passes_every_test_even_when_it_prints():
    assert outcomes('def add(a, b):\n    return a + b\nprint(add(2, 2))\nif __name__ == "__main__":\n    print("demo")\n') == [True, True]


def test_partial_and_wrong_candidates_get_partial_or_no_credit():
    assert outcomes('def add(a, b):\n    return abs(a) + abs(b)\n') == [True, False]
    assert outcomes('def add(a, b):\n    return 0\n') == [False, False]


@pytest.mark.parametrize('cheat', [
    'print("A1b2C3d4E5f6G7h8")\n',                      # constant guess captured by the throwaway stdout
    'import os\nos.write(1, b"OK\\n")\nos._exit(0)\n',   # bypass stdout, exit before the tests
    'import sys\nsys.exit(0)\n',
    'x = input()\n',                                      # hidden program is not readable after the prefix
])
def test_candidates_cannot_pass_without_running_the_hidden_tests(cheat):
    assert outcomes(cheat) == [False, False]


def test_reward_bridge_wraps_only_function_tasks(tmp_path):
    import json
    from posttrain.common import digest
    from posttrain.execution.contracts import CaseResult, JudgeResult, Status
    from posttrain.execution.function_harness import PREFIX
    from posttrain.training.verl_pilot_reward import score_candidate
    sent = []
    class Fixture:
        def score(self, problem_id, code, request_id):
            sent.append(code)
            return JudgeResult(request_id, Status.OK, 2, 2, (CaseResult(Status.OK),) * 2, .1, 0., 2, 0, False,
                               'fixture', 'isolated_execution', digest(code), 'h')
    base = {'tests_hash': 'h', 'test_count': 2}
    for i, (pid, wrapped) in enumerate((('kodcode-Evol_1', True), ('taco-train-000001', False))):
        score_candidate('```python\ndef add(a, b):\n    return a + b\n```', {**base, 'problem_id': pid},
                        Fixture(), tmp_path / pid, f'{i}' * 32, 'binary')
        assert sent[-1].startswith(PREFIX) is wrapped
        record = json.loads((tmp_path / pid / 'records' / (f'{i}' * 32 + '.json')).read_text(encoding='utf-8'))
        assert record['code_hash'] == digest(sent[-1])


def test_program_shape_and_identity():
    pairs = hidden_programs(TESTS, TOKEN)
    assert len(pairs) == 2 and all('from solution' not in s for s, _ in pairs)
    assert pairs[0][0].rstrip().endswith("print('A1b2C3d4E5f6G7h8')") and 'test_small()' in pairs[0][0]
    assert is_function_task('kodcode-Evol_1') and not is_function_task('taco-train-000001')
    with pytest.raises(ValueError):
        hidden_programs('def helper():\n    pass\n', TOKEN)
    with pytest.raises(ValueError):
        hidden_programs(TESTS, 'short')
