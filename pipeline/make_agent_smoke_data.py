"""Smoke-test data for phase-2 Agent RL (not a training pool for results): the phase-1 extension pool
(data/prepared/kodcode-rl-v2, 138 KodCode tasks already loaded in the scoring gateway) in both single and
Agent format (Agent prompts via agent_messages -> SYSTEM_FUNCTION_V5). Output: data/prepared/kodcode-agent-smoke.
"""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('prep', ROOT/'scripts/prepare_rl_pool.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
from posttrain.agent.episode import agent_messages  # noqa: E402
from posttrain.common import file_digest, write_json  # noqa: E402

SRC = ROOT/'data/prepared/kodcode-rl-v2-source'
OUT = ROOT/'data/prepared/kodcode-agent-smoke'


def main():
    source = OUT.with_name(OUT.name + '-source')
    if source.exists() or OUT.exists():
        raise SystemExit('output exists')
    source.mkdir(parents=True)
    for name in ('single-train.jsonl', 'single-development.jsonl'):
        (source/name).write_bytes((SRC/name).read_bytes())
    with open(source/'agent-train.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for line in (SRC/'single-train.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            f.write(json.dumps({**r, 'messages': agent_messages(r['messages'])}, ensure_ascii=False) + '\n')
    (source/'agent-development.jsonl').write_bytes((ROOT/'data/prepared/agent-rl-v5b/agent-development.jsonl').read_bytes())
    names = ('single-train.jsonl', 'single-development.jsonl', 'agent-train.jsonl', 'agent-development.jsonl')
    write_json(source/'manifest.json', {'hidden_tests_exported': False, 'scope': 'kodcode_public_prompts_smoke_only',
        'files': {n: file_digest(source/n) for n in names}})
    prep.ROOT = ROOT
    print(json.dumps(prep.prepare(source, ROOT/'data/prepared/kodcode-rl-v2/rl-problem-pool.json', OUT, ['single', 'agent'])))


if __name__ == '__main__':
    main()
