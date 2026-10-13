from __future__ import annotations

import random
import statistics
from collections import defaultdict


def quantile(values: list[float], p: float) -> float:
    values=sorted(values)
    index=(len(values)-1)*p
    lo=int(index)
    frac=index-lo
    return values[lo]*(1-frac)+values[min(lo+1,len(values)-1)]*frac


def paired_report(rows: list[dict], bootstrap: int = 10000, seed: int = 20260925,
                  cluster_resampling: bool = False) -> dict:
    """One row/task with both paired seeds; identical sampled indices for all runs."""
    if not rows or len({r['task_id'] for r in rows})!=len(rows):
        raise ValueError('empty_or_duplicate_tasks')
    columns=['m2_42','m3_42','m2_43','m3_43']
    if any(type(r.get(k)) is not bool for r in rows for k in columns):
        raise ValueError('every_task_requires_valid_boolean_outcomes_for_all_four_runs')
    if bootstrap<1:
        raise ValueError('invalid_bootstrap_count')
    strata=defaultdict(lambda:defaultdict(list))
    cluster_levels={}
    for i,row in enumerate(rows):
        level=row.get('difficulty','all')
        key=row['cluster_id'] if cluster_resampling else row['task_id']
        if not key:
            raise ValueError('missing_cluster_id')
        if key in cluster_levels and cluster_levels[key]!=level:
            raise ValueError('cluster_spans_strata_freeze_a_cluster_stratum_first')
        cluster_levels[key]=level
        strata[level][key].append(i)
    effects=[((r['m3_42']-r['m2_42'])+(r['m3_43']-r['m2_43']))*50 for r in rows]
    rng=random.Random(seed)
    samples=[]
    for _ in range(bootstrap):
        indices=[]
        for level in sorted(strata):
            units=list(strata[level].values())
            for _ in range(len(units)):
                indices.extend(rng.choice(units))
        samples.append(statistics.mean(effects[i] for i in indices))
    differences={str(s):100*statistics.mean(r[f'm3_{s}']-r[f'm2_{s}'] for r in rows) for s in (42,43)}
    mean=statistics.mean(differences.values())
    ci=[quantile(samples,.025),quantile(samples,.975)]
    return {'tasks':len(rows),'greedy_pass1':{k:statistics.mean(r[k] for r in rows) for k in columns},
            'paired_difference_pp':differences,'mean_difference_pp':mean,'bootstrap_95_percent_pp':ci,
            'bootstrap_seed':seed,'bootstrap_replicates':bootstrap,'cluster_resampling':cluster_resampling,
            'interval_includes_zero':ci[0]<=0<=ci[1],
            'interpretation':'no_average_improvement' if mean<=0 else 'both_seeds_positive' if all(v>0 for v in differences.values()) else 'unstable',
            'uncertainty_scope':'task_sampling_conditional_on_these_training_runs_not_training_randomness'}
