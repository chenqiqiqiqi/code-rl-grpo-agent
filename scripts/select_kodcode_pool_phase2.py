"""Phase-2 pool (C and A share it): shard-2 problems that are 0<p<1 (k=4) for W in single-turn OR Agent
screening (union), minus an optional exclusion list (e.g. the stricter decontamination recheck), original
order; builds a prepared dir with both `single` and `agent` modes. Agent prompts are the single-turn prompts
with the current Agent protocol system prompt (posttrain.agent.episode.agent_messages -> SYSTEM_FUNCTION_V5).
Usage: python scripts/select_kodcode_pool_phase2.py --single-records DIR --agent-records DIR
         [--exclude ids.json] --output data/prepared/kodcode-rl-v3
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('v2', ROOT/'scripts/select_kodcode_pool_v2.py')
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
from posttrain.agent.episode import agent_messages  # noqa: E402
from posttrain.common import file_digest, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--single-records', type=Path, required=True)
    p.add_argument('--agent-records', type=Path, required=True)
    p.add_argument('--exclude', type=Path)
    p.add_argument('--prompts', type=Path, default=ROOT/'data/kodcode-rl-v2-shard2/gpu-phase2-prompts.jsonl')
    p.add_argument('--development', type=Path, default=ROOT/'data/prepared/rl-pool-v5b/single-development.jsonl')
    p.add_argument('--agent-development', type=Path, default=ROOT/'data/prepared/agent-rl-v5b/agent-development.jsonl')
    p.add_argument('--output', type=Path, default=ROOT/'data/prepared/kodcode-rl-v3')
    a = p.parse_args()
    lines = a.prompts.read_text(encoding='utf-8').splitlines()
    rows = [json.loads(l) for l in lines]
    order = [r['problem_id'] for r in rows]
    single, agent = v2.rates(a.single_records), v2.rates(a.agent_records)
    if set(single) != set(order) or set(agent) != set(order):
        raise SystemExit('screening does not cover exactly the prompt set')
    excluded = set(json.loads(a.exclude.read_text(encoding='utf-8'))) if a.exclude else set()
    if excluded - set(order):
        raise SystemExit('exclusion list has ids outside the prompt set')
    chosen = [pid for pid in order if pid not in excluded and (0 < single[pid] < 1 or 0 < agent[pid] < 1)]
    stats = {'screened': len(order), 'excluded': len(excluded), 'pool': len(chosen),
             'mid': {'single': sum(0 < single[x] < 1 for x in order), 'agent': sum(0 < agent[x] < 1 for x in order)},
             'all_pass': {'single': sum(single[x] == 1 for x in order), 'agent': sum(agent[x] == 1 for x in order)},
             'all_fail': {'single': sum(single[x] == 0 for x in order), 'agent': sum(agent[x] == 0 for x in order)}}
    source = a.output.with_name(a.output.name + '-source')
    if source.exists() or a.output.exists():
        raise SystemExit('output exists')
    source.mkdir(parents=True)
    (source/'single-train.jsonl').write_bytes(a.prompts.read_bytes())
    (source/'single-development.jsonl').write_bytes(a.development.read_bytes())
    with open(source/'agent-train.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps({**r, 'messages': agent_messages(r['messages'])}, ensure_ascii=False) + '\n')
    (source/'agent-development.jsonl').write_bytes(a.agent_development.read_bytes())
    names = ('single-train.jsonl', 'single-development.jsonl', 'agent-train.jsonl', 'agent-development.jsonl')
    write_json(source/'manifest.json', {'hidden_tests_exported': False, 'scope': 'kodcode_public_prompts',
        'files': {n: file_digest(source/n) for n in names}})
    write_json(source/'rl-problem-pool.json', {'rule': '0<p<1 at k=4 for W single-turn or W Agent (union), minus exclusions',
        'records': {'single': str(a.single_records), 'agent': str(a.agent_records)},
        'exclude': str(a.exclude) if a.exclude else None, 'stats': stats,
        'pool': [{'problem_id': pid, 'p_single': single[pid], 'p_agent': agent[pid]} for pid in chosen]})
    v2.prep.ROOT = ROOT
    v2.prep.prepare(source, source/'rl-problem-pool.json', a.output, ['single', 'agent'])
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
