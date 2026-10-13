"""Held-out training probe set comparison (pre-registered rule in STAGE4_PHASE1_PLAN_DRAFT.md).

Usage: python stage4-phase1/compare_probe.py
A probe counts as a clear improvement when its paired 95% bootstrap CI against M0 (both k=8, T=0.7,
same 160 problems) has a lower bound > 0 and format rejections did not jump; among several, take the
largest point estimate (close calls go to the configuration with fewer changes, probe1 before probe2).
Binary@169 and mixed@90 rows are context only: those runs had already trained on these problems.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ab', ROOT/'eval-baseline-20260930/analyze_baseline.py')
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)


def load(name):
    folder = ROOT/f'stage4-phase1/probe/{name}-probe-s42'
    if not (folder/'records').is_dir():
        return None
    by = ab.load_recs(folder)
    rate, _ = ab.summarize_single(by, name)
    rejected = sum(r['format_rejected'] for v in by.values() for r in v)
    return rate, rejected


def main():
    m0 = load('m0')
    if not m0:
        raise SystemExit('M0 probe-set records missing')
    verdict = {}
    for name in ('binary169', 'mixed90', 'probe1', 'probe2'):
        got = load(name)
        if not got:
            continue
        lo, _ = ab.paired(got[0], m0[0], 8, 8, f'{name} - M0')
        if name.startswith('probe'):
            diff = sum(got[0].values()) - sum(m0[0].values())
            ok = lo > 0 and got[1] <= max(2 * m0[1], m0[1] + 20)
            verdict[name] = (ok, diff)
    print('verdict:', {k: ('CLEAR IMPROVEMENT' if ok else 'no clear improvement', round(d, 2)) for k, (ok, d) in verdict.items()})
    winners = [k for k, (ok, _) in verdict.items() if ok]
    if winners:
        best = max(winners, key=lambda k: (round(verdict[k][1]), k == 'probe1'))
        print('RULE: rerun phase 1 with', best)
    elif len(verdict) == 2:
        print('RULE: no probe clearly improved -> diagnose cause (signal density vs learning), then KodCode or probe3')


if __name__ == '__main__':
    main()
