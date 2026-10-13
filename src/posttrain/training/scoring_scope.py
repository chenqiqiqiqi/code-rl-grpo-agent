"""Separate full development selection from a frozen training-only target filter."""

def checked_scoring_scope(prompts,prompts_sha256,training_manifest=None):
    if training_manifest is None:
        if len(prompts)!=348 or {p['split'] for p in prompts}!={'development'}:
            raise ValueError('full_348_development_prompts_required')
        return 'development'
    ids=[p['problem_id'] for p in prompts]
    if (training_manifest.get('training_only') is not True or
        training_manifest.get('prompts_sha256')!=prompts_sha256 or
        training_manifest.get('cohort_ids')!=ids or len(ids)!=512 or len(set(ids))!=512 or
        {p['split'] for p in prompts}!={'train'}):
        raise ValueError('frozen_512_training_only_filter_required')
    return 'training_target_filter'
