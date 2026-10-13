"""Deterministic candidate allocation, with explicit unresolved data-quality gates."""
from posttrain.data.sampling import split_clusters, stratified_sample
from posttrain.common import file_digest


def verify_bound_files(root, hashes):
    root=root.resolve()
    for name,expected in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('artifact_path_outside_root')
        if not path.is_file() or file_digest(path)!=expected:
            raise ValueError('artifact_hash_mismatch:'+name)


def admission_split(problems, review, accepted_sha256, calibration_size=200):
    problems=sorted(problems,key=lambda p:p.problem_id)
    ids={p.problem_id for p in problems}
    if len(ids)!=len(problems):raise ValueError('unique_problem_ids_required')
    if any(p.validation_status!='reference_passed' or len(p.tests)<5 for p in problems):
        raise ValueError('validated_reference_and_five_tests_required')
    if review.get('source_accepted_sha256')!=accepted_sha256:
        raise ValueError('review_source_changed')
    if review.get('human_review_required') is not False:
        raise ValueError('explicit_human_review_waiver_required')
    bad=set(review['quarantine_ids']); checked=set(review['domain_valid_ids'])
    if not bad|checked<=ids or bad&checked:
        raise ValueError('review_ids_inconsistent')
    bad_clusters={p.cluster_id for p in problems if p.problem_id in bad}
    retained=[p for p in problems if p.cluster_id not in bad_clusters]
    split=split_clusters(retained)
    train=set(split['train_ids']);dev=set(split['dev_ids'])
    calibration=stratified_sample([{'task_id':p.problem_id,'difficulty':p.difficulty}
                                   for p in retained if p.problem_id in train],calibration_size)
    unseen={p.problem_id for p in retained}-checked
    split.update(
        status='candidate_allocation_only_pending_data_and_GPU_gates',
        quarantine_policy='whole_cluster_no_frozen_test_edits',
        quarantined_ids=sorted(ids-{p.problem_id for p in retained}),
        reviewed_domain_valid_ids=sorted(checked & {p.problem_id for p in retained}),
        domain_unchecked_ids=sorted(unseen),domain_unchecked_count=len(unseen),
        data_domain_gate_passed=not unseen,
        sft_ids=sorted(train),rl_ids=sorted(train),SFT_RL_overlap=len(train),
        calibration_seed=20260925,calibration_ids=[r['task_id'] for r in calibration],
        human_review_required=False,human_review_complete=False,
        training_ready=False,protocol_freeze_complete=False,
        cluster_overlap=len({p.cluster_id for p in retained if p.problem_id in train} &
                            {p.cluster_id for p in retained if p.problem_id in dev}),
    )
    return split
