"""Unseal the KodCode final test set for the one-time final evaluation (protocol v4 addenda 4/9/10).

Builds the public GPU prompts (system prompt + statement; no tests, no tokens, no references) from the
verified VM problem file exactly like finalize_dev in scripts/prepare_kodcode_rl.py, and records the unsealing
in data/kodcode-rl-v1/test/unseal.json (finalize.json is left untouched). The `split` field must be one of the
prompt-manifest splits accepted by checked_prompt; it is set to 'development' as a format constraint only —
these are the sealed test problems.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from posttrain.common import file_digest, utc_now, write_json, write_jsonl  # noqa: E402
from posttrain.data.prompt_manifest import checked_prompt  # noqa: E402
from posttrain.data.schema import Problem  # noqa: E402
from posttrain.execution.function_harness import SYSTEM_PROMPT  # noqa: E402

TEST = ROOT/'data/kodcode-rl-v1/test'
SEALED_SHA256 = '39c2d9c7e06d19b0e7c0d4368e549c2ef536d1aebbee56f767372e58821cb348'


def main():
    src, out = TEST/'vm-test-problems.jsonl', TEST/'gpu-test-prompts.jsonl'
    if file_digest(src) != SEALED_SHA256:
        raise SystemExit('sealed test file changed')
    if out.exists():
        raise SystemExit('already unsealed')
    prompts = []
    for line in src.read_text(encoding='utf-8').splitlines():
        p = Problem.from_dict(json.loads(line))
        if 'def test_' in p.statement:
            raise SystemExit('statement quotes a test: ' + p.problem_id)
        prompts.append(checked_prompt({'problem_id': p.problem_id,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': p.statement}],
            'tests_hash': p.tests_hash, 'test_count': len(p.tests), 'difficulty': p.difficulty,
            'cluster_id': p.cluster_id, 'split': 'development'}))
    write_jsonl(out, prompts)
    write_json(TEST/'unseal.json', {'unsealed_at': utc_now(), 'reason': 'one-time final evaluation (KodCode test only; user 2026-10-02)',
        'problems': len(prompts), 'files': {'vm-test-problems.jsonl': SEALED_SHA256, 'gpu-test-prompts.jsonl': file_digest(out)}})
    print(json.dumps({'prompts': len(prompts)}))


if __name__ == '__main__':
    main()
