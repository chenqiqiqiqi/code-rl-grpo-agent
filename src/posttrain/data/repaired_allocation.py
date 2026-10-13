"""Provisional split of an independently verified fresh eligible pool.

Capacity checks do not replace boundary review, protocol freeze or GPU pilot.
The caller is responsible for authenticating the fresh evidence and eligibility.
"""
from posttrain.data.sampling import split_clusters, stratified_sample


def allocate_verified_problems(problems):
    problems = sorted(problems, key=lambda p: p.problem_id)
    if len({p.problem_id for p in problems}) != len(problems):
        raise ValueError('duplicate_eligible_problem')
    if any(p.validation_status != 'reference_passed' or len(p.references) != 1
           or len(p.tests) < 5 or not p.cluster_id for p in problems):
        raise ValueError('fresh_reference_cluster_and_five_tests_required')
    enough_clusters = len({p.cluster_id for p in problems}) >= 2
    split = split_clusters(problems) if enough_clusters else {
        'train_ids': [], 'dev_ids': [], 'cluster_by_id': {},
        'seed': 20260925, 'dev_fraction': .15}
    train = set(split['train_ids'])
    dev = set(split['dev_ids'])
    calibration = stratified_sample([
        {'task_id': p.problem_id, 'difficulty': p.difficulty}
        for p in problems if p.problem_id in train], 200) if len(train) >= 200 else []
    overlap = ({p.cluster_id for p in problems if p.problem_id in train}
               & {p.cluster_id for p in problems if p.problem_id in dev})
    if overlap or train & dev:
        raise ValueError('cluster_or_problem_split_leakage')
    split.update({
        'scope': 'provisional_allocation_not_training_authorization',
        'split_complete': enough_clusters,
        'split_blocker': None if enough_clusters else 'need_at_least_two_clusters',
        'eligible_problems': len(problems),
        'calibration_requested': 200,
        'calibration_ids': [r['task_id'] for r in calibration],
        'calibration_seed': 20260925,
        'cluster_overlap': 0,
        'sft_ids': sorted(train),
        'rl_candidate_ids': sorted(train),
        'rl_sampling_frozen': False,
        'protocol_capacity_checks': {
            'sft_capacity_1500_to_3000': 1500 <= len(train) <= 3000,
            'rl_capacity_at_least_2000': len(train) >= 2000,
            'development_300_to_500': 300 <= len(dev) <= 500,
            'calibration_200_available_in_train': len(calibration) == 200,
        },
        'boundary_sample_review_complete': False,
        'training_ready': False,
        'protocol_freeze_complete': False,
    })
    return split
