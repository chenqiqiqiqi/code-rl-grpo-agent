"""Pre-registered B1 rule (protocol v4 Stage 4 item 3), run before any formal branch starts.

Mean of min(len, 1024) over the single-turn verl pilot's T=1.0 responses; if it differs
from 325 by more than 20%, B1 = 5548 x measured mean (rounded), else B1 = 1,800,000.
Usage: python b1_from_pilot.py runs/verl-4090d-20260930/single
"""
import glob
import json
import statistics
import sys

CAP, ASSUMED, TRAJECTORIES, DEFAULT = 1024, 325, 5548, 1_800_000

lengths = []
for path in glob.glob(sys.argv[1] + '/audit/rollouts/*.json'):
    for c in json.load(open(path, encoding='utf-8'))['candidates']:
        lengths.append(min(len(c['response_tokens']), CAP))
if not lengths:
    raise SystemExit('no pilot rollouts found')
mean = statistics.fmean(lengths)
b1 = DEFAULT if abs(mean - ASSUMED) <= 0.2 * ASSUMED else round(TRAJECTORIES * mean)
print(json.dumps({'pilot_responses': len(lengths), 'mean_min_len_1024': round(mean, 1),
                  'share_at_or_over_1024': round(sum(x >= CAP for x in lengths) / len(lengths), 4),
                  'assumed_mean': ASSUMED, 'adjusted': b1 != DEFAULT, 'B1': b1}))
