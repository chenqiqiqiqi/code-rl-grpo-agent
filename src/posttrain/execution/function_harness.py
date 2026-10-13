"""Function-task (KodCode) harness on the unchanged stdio scorer.

Each frozen TestCase of a function task carries one hidden assert-test program as stdin and a
per-problem secret token as expected stdout. The public prefix reads the hidden program before the
candidate runs, then gives the candidate an empty stdin and a throwaway stdout; the suffix restores
stdout and executes the hidden program in the candidate's namespace. The token is printed only after
the test's asserts pass, so a candidate cannot pass by printing a constant. Hidden programs and tokens
stay on the CPU scorer; the GPU side only wraps candidate code with this public prefix/suffix.
"""
from __future__ import annotations

import re

FUNCTION_TASK_PREFIX = 'kodcode-'
SYSTEM_PROMPT = ('Write Python 3 code that implements the requested function(s). '
                 'Do not read input or print results. Return code only.')
PREFIX = ('import sys as _kc_sys, io as _kc_io\n'
          '_kc_hidden = _kc_sys.stdin.read()\n'
          '_kc_sys.stdin = _kc_io.StringIO("")\n'
          '_kc_out = _kc_sys.stdout\n'
          '_kc_sys.stdout = _kc_io.StringIO()\n')
SUFFIX = ('\n_kc_sys.stdout = _kc_out\n'
          'exec(compile(_kc_hidden, "<hidden-tests>", "exec"), globals())\n')
TEST_NAME = re.compile(r'(?m)^def (test_\w+)\(\s*\)\s*:')
SOLUTION_IMPORT = re.compile(r'(?m)^\s*(?:from\s+solution\s+import\s+.*|import\s+solution\s*)$')


def is_function_task(problem_id: str) -> bool:
    return problem_id.startswith(FUNCTION_TASK_PREFIX)


def wrap(code: str) -> str:
    if not code.endswith('\n'):
        code += '\n'
    return PREFIX + code + SUFFIX


def test_names(test_code: str) -> list[str]:
    names = TEST_NAME.findall(test_code)
    if not names or len(set(names)) != len(names):
        raise ValueError('function_tests_need_unique_plain_test_functions')
    return names


def hidden_programs(test_code: str, token: str) -> list[tuple[str, str]]:
    """One (stdin, expected stdout) pair per test function; all definitions, one call each."""
    if not re.fullmatch(r'[A-Za-z0-9]{16,64}', token):
        raise ValueError('invalid_secret_token')
    body = SOLUTION_IMPORT.sub('', test_code).rstrip() + '\n'
    return [(body + f'\n{name}()\nprint({token!r})\n', token + '\n') for name in test_names(body)]
