"""Collapse rule (protocol v4 addendum 2): actor entropy < 0.15 for 20 consecutive logged steps.

Usage (GPU host): python entropy_check.py RUN_LOG   -> prints last entropy and ALERT when the rule fires.
"""
import re
import sys

values = {}
for line in open(sys.argv[1], errors='ignore'):
    m = re.search(r'step:(\d+) - .*?actor/entropy:(?:np\.float\d+\()?(-?[\d.]+(?:e-?\d+)?)', line)
    if m:
        values[int(m.group(1))] = float(m.group(2))
steps = sorted(values)
run = 0
for s in steps:
    run = run + 1 if values[s] < 0.15 else 0
last = f'{values[steps[-1]]:.3f}@{steps[-1]}' if steps else 'none'
print(f'entropy {last}' + (' ALERT entropy<0.15 for 20 steps' if run >= 20 else ''))
