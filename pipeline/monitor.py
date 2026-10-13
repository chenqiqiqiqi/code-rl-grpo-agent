"""Per-step summary of a formal code_rl run and the pre-registered engineering abort rules.

Pure Python (no torch); reads the run's audit trail only. Usage: python monitor.py RUN_DIR [--last N]
Abort rules (protocol v4 Stage 4 item 5): rollout-vs-actor mean |dlogp| > 0.02 for 3
consecutive steps; format-rejection share > 50% for 20 consecutive steps.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def steps(run):
    audit = run/'audit'
    olds = {}
    for p in (audit/'old_log_probs').glob('*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        olds[(d['session'], d['step'])] = {r['request_id']: r['old_log_probs'] for r in d['rows']}
    rows = []
    for p in (audit/'rollouts').glob('*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        cands = d['candidates']
        old = olds.get((d['session'], d['step']))
        diffs = []
        if old:
            for c in cands:
                diffs += [abs(a - b) for a, b in zip(c['rollout_log_probs'], old[c['request_id']], strict=True)]
        rejected = 0
        for c in cands:
            rec = run/'scoring/records'/(c['request_id'] + '.json')
            if rec.is_file() and json.loads(rec.read_text(encoding='utf-8')).get('format_rejected'):
                rejected += 1
        groups = {}
        for c in cands:
            groups.setdefault(c['uid'], []).append(c['mixed'] if c['mixed'] is not None else c['binary'])
        lengths = [len(c['response_tokens']) for c in cands]
        rows.append({'session': d['session'][:6], 'step': d['step'], 'n': len(cands),
                     'success': sum(c['binary'] == 1.0 for c in cands),
                     'mixed_mean': (statistics.fmean(c['mixed'] for c in cands)
                                    if all(c['mixed'] is not None for c in cands) else None),
                     'signal_groups': sum(len(set(v)) > 1 for v in groups.values()), 'groups': len(groups),
                     'format_rejected': rejected, 'len_mean': statistics.fmean(lengths),
                     'at_cap': sum(x >= 1024 for x in lengths), 'tokens': sum(lengths),
                     'dlogp_mean': statistics.fmean(diffs) if diffs else None,
                     'wall_s': d['generation_and_scoring_wall_s']})
    return sorted(rows, key=lambda r: (r['step'], r['session']))


def ledger(run):
    db = sqlite3.connect((run/'audit/tokens.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        cap = db.execute('SELECT cap FROM config').fetchone()[0]
        used, pending = db.execute('SELECT COALESCE(SUM(COALESCE(actual,worst)),0), SUM(actual IS NULL) '
                                   'FROM reservations').fetchone()
        return {'cap': cap, 'charged': used, 'remaining': cap - used, 'unsettled_reservations': pending}
    finally:
        db.close()


def rules(rows):
    alerts, run_d, run_f = [], 0, 0
    for r in rows:
        run_d = run_d + 1 if r['dlogp_mean'] is not None and r['dlogp_mean'] > 0.02 else 0
        run_f = run_f + 1 if r['format_rejected'] > 0.5 * r['n'] else 0
        if run_d >= 3: alerts.append(f"step {r['step']}: dlogp>0.02 for 3 steps (sync failure rule)")
        if run_f >= 20: alerts.append(f"step {r['step']}: format rejection >50% for 20 steps (LR rule)")
    return alerts


def main():
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); p.add_argument('--last', type=int, default=15)
    a = p.parse_args()
    rows = steps(a.run)
    for r in rows[-a.last:]:
        mixed = '' if r['mixed_mean'] is None else f" mixed={r['mixed_mean']:.3f}"
        dl = 'n/a' if r['dlogp_mean'] is None else f"{r['dlogp_mean']:.4f}"
        print(f"{r['session']} step {r['step']:4d}  succ {r['success']:2d}/{r['n']}{mixed}  signal {r['signal_groups']}/{r['groups']}"
              f"  rej {r['format_rejected']:2d}  len {r['len_mean']:6.1f} cap {r['at_cap']:2d}  dlogp {dl}  wall {r['wall_s']:.0f}s")
    completion = a.run/'audit/completion.json'
    print(json.dumps({'steps_logged': len(rows), 'ledger': ledger(a.run),
                      'completion': json.loads(completion.read_text()) if completion.is_file() else None,
                      'alerts': rules(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
