"""Paired comparison of scored candidate sets against a baseline (same problems, per-problem rates).

Usage: python stage4-phase1/compare_any.py BASELINE_DIR[,BASELINE_DIR2] NAME=DIR[,DIR2] [NAME=DIR ...]
Each DIR holds VM scoring `records/`; comma-separated dirs (baseline or named) are pooled per problem
(e.g. seed 42 + 43 for k=16). Reuses the baseline analysis' summary, paired bootstrap and sign test.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ab', ROOT/'eval-baseline-20260930/analyze_baseline.py')
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)


def rates(folder, name):
    return ab.summarize_single(ab.load_recs(Path(folder)), name)


def main():
    bases = [rates(d, f'baseline {Path(d).name}') for d in sys.argv[1].split(',')]
    base = {p: sum(r[0][p] for r in bases) / len(bases) for p in bases[0][0]}
    kb = sum(r[1] for r in bases)
    print(f'[baseline pooled k={kb}] expected passes {sum(base.values()):.2f}')
    for arg in sys.argv[2:]:
        name, folders = arg.split('=', 1)
        parts = [rates(d, f'{name} {Path(d).name}') for d in folders.split(',')]
        rate = {p: sum(r[0][p] for r in parts) / len(parts) for p in parts[0][0]}
        k = sum(r[1] for r in parts)
        if len(parts) > 1:
            print(f'[{name} pooled k={k}] expected passes {sum(rate.values()):.2f}')
        ab.paired(rate, base, k, kb, f'{name} - baseline')


if __name__ == '__main__':
    main()
