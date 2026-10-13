from __future__ import annotations

import math
import random
from collections import defaultdict

from posttrain.common import digest
from posttrain.data.schema import Problem


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or any(v < 0 for v in counts.values()) or sum(counts.values()) < total:
        raise ValueError("insufficient_candidates")
    denominator = sum(counts.values())
    if denominator == 0:
        return {k:0 for k in counts}
    # Integer arithmetic gives deterministic ties, independent of float rounding.
    out = {k:v*total//denominator for k,v in counts.items()}
    ranked = sorted(counts,key=lambda k:(-(counts[k]*total % denominator),k))
    for k in ranked[:total-sum(out.values())]:
        out[k] += 1
    return out


def stratified_sample(rows: list[dict], total: int, seed: int = 20260925) -> list[dict]:
    if len({x['task_id'] for x in rows}) != len(rows):
        raise ValueError("duplicate_task_ids")
    groups = defaultdict(list)
    for row in rows:
        groups[row['difficulty']].append(row)
    quotas = largest_remainder({k:len(v) for k,v in groups.items()}, total)
    rng = random.Random(seed)
    result = []
    for level in sorted(groups):
        result.extend(rng.sample(sorted(groups[level],key=lambda r:r['task_id']),quotas[level]))
    return sorted(result,key=lambda r:r['task_id'])


def split_clusters(problems: list[Problem], dev_fraction: float = .15, seed: int = 20260925) -> dict:
    if not 0 < dev_fraction < 1 or any(not p.cluster_id for p in problems):
        raise ValueError("invalid_split_configuration")
    groups = defaultdict(list)
    for p in problems:
        groups[p.cluster_id].append(p.problem_id)
    keys = sorted(groups, key=lambda k:digest([seed,k]))
    if len(keys) < 2:
        raise ValueError("need_at_least_two_clusters")
    dev_keys = set(keys[:min(len(keys)-1,max(1,math.ceil(len(keys)*dev_fraction)))])
    return {"seed":seed,"dev_fraction":dev_fraction,
            "train_ids":sorted(p.problem_id for p in problems if p.cluster_id not in dev_keys),
            "dev_ids":sorted(p.problem_id for p in problems if p.cluster_id in dev_keys),
            "cluster_by_id":{p.problem_id:p.cluster_id for p in problems}}
