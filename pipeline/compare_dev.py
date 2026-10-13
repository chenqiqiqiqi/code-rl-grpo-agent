"""Stage 4 phase 1 development comparison (pre-registered rule, EXPERIMENT_PROTOCOL v4 Stage 4 item 6).

Usage: python stage4-phase1/compare_dev.py
Reads VM scoring records from stage4-phase1/dev/records/{binary,mixed}-s{42,43} and the M0 baseline
from eval-baseline-20260930/records/single-s{42,43}; reuses the baseline's paired bootstrap / sign test.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ab', ROOT/'eval-baseline-20260930/analyze_baseline.py')
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)


def rates(folder, name):
    return ab.summarize_single(ab.load_recs(folder), name)[0] if (folder/'records').is_dir() else None


def pooled(*rs):
    rs = [r for r in rs if r]
    return {p: sum(r[p] for r in rs) / len(rs) for p in rs[0]} if rs else None


def main():
    m0 = pooled(*(rates(ROOT/f'eval-baseline-20260930/records/single-s{s}', f'M0 seed {s}') for s in (42, 43)))
    run = {(b, s): rates(ROOT/f'stage4-phase1/dev/records/{b}-s{s}', f'{b} seed {s}')
           for b in ('binary', 'mixed') for s in (42, 43)}
    print(f'[M0 pooled k=16] expected passes {sum(m0.values()):.2f}')
    for b in ('binary', 'mixed'):
        r = pooled(run[(b, 42)], run[(b, 43)])
        if r:
            k = 16 if run[(b, 43)] else 8
            print(f'[{b} k={k}] expected passes {sum(r.values()):.2f}')
            ab.paired(r, m0, k, 16, f'{b} (k={k}) - M0 (k=16)')
    if run[('binary', 42)] and run[('mixed', 42)]:
        delta = sum(run[('mixed', 42)].values()) - sum(run[('binary', 42)].values())
        ab.paired(run[('mixed', 42)], run[('binary', 42)], 8, 8, 'mixed - binary (seed 42)')
        if abs(delta) <= 2 and not (run[('binary', 43)] and run[('mixed', 43)]):
            print(f'RULE: |delta| = {abs(delta):.2f} <= 2 -> add seed 43 for both branches, decide on k=16')
            return
        if abs(delta) <= 2:
            b16, m16 = pooled(run[('binary', 42)], run[('binary', 43)]), pooled(run[('mixed', 42)], run[('mixed', 43)])
            delta = sum(m16.values()) - sum(b16.values())
            ab.paired(m16, b16, 16, 16, 'mixed - binary (k=16)')
        print(f'RULE: delta = {delta:+.2f} -> W = {"mixed" if delta > 0 else "binary"}')


if __name__ == '__main__':
    main()
