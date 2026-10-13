"""Held-out training-split probe set for the low-cost LR/G probes (no development data).

The seed-42 sampler permutation does not depend on batch size, so the binary run's rollouts give
the visiting order: a probe of S steps x B prompts only sees the first S*B problems. Pick N problems
from later positions whose M0 screening single-turn rate was > 0, lines byte-identical to the pool.
Usage (GPU host): python make_probe_set.py [first_unseen_position=320] [N=160]
"""
import json, glob, random, sys
from pathlib import Path

P = Path('/root/autodl-tmp/posttrain-project')
FIRST, N = (int(sys.argv[1]) if len(sys.argv) > 1 else 320), (int(sys.argv[2]) if len(sys.argv) > 2 else 160)
pool = {p['problem_id']: p for p in json.load(open(P/'data/prepared/rl-pool-v5b/rl-problem-pool.json'))['pool']}
steps = {}
for f in glob.glob(str(P/'runs/stage4-phase1/binary/audit/rollouts/*.json')):
    d = json.load(open(f))
    ids = list(dict.fromkeys(c['problem_id'] for c in d['candidates']))  # group order within the step
    steps.setdefault(d['step'], []).append(ids)
if any(len(v) != 1 for v in steps.values()):
    raise SystemExit('binary run has repeated steps; visiting order ambiguous')
order = [pid for s in sorted(steps) for pid in steps[s][0]]
seen = set(order[:FIRST])
eligible = sorted(pid for pid in order[FIRST:] if pool[pid]['single_p'] > 0 and pid not in seen)
chosen = set(random.Random(0).sample(eligible, N))
lines = [l for l in open(P/'data/prepared/rl-pool-v5b/single-train.jsonl', 'rb') if json.loads(l)['problem_id'] in chosen]
out = P/'data/prepared/rl-pool-v5b-probe'; out.mkdir(exist_ok=False)
(out/'probe-train.jsonl').write_bytes(b''.join(lines))
json.dump({'first_unseen_position': FIRST, 'eligible': len(eligible), 'chosen': sorted(chosen),
           'visited_order_head': order[:FIRST]}, open(out/'probe-set.json', 'w'), indent=1)
print(json.dumps({'visited_positions': len(order), 'eligible': len(eligible), 'chosen': len(lines),
                  'mean_m0_single_p': round(sum(pool[p]['single_p'] for p in chosen) / N, 3)}))
