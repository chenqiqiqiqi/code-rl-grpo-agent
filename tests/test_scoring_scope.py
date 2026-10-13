import pytest
from posttrain.training.scoring_scope import checked_scoring_scope

def rows(n,split):
    return [{'problem_id':str(i),'split':split} for i in range(n)]

def test_development_scope_cannot_silently_admit_training():
    assert checked_scoring_scope(rows(348,'development'),'abc')=='development'
    with pytest.raises(ValueError):checked_scoring_scope(rows(512,'train'),'abc')

def test_train_filter_requires_frozen_cohort_and_hash():
    prompts=rows(512,'train')
    manifest={'training_only':True,'prompts_sha256':'abc','cohort_ids':[r['problem_id'] for r in prompts]}
    assert checked_scoring_scope(prompts,'abc',manifest)=='training_target_filter'
    with pytest.raises(ValueError):checked_scoring_scope(prompts,'changed',manifest)
    with pytest.raises(ValueError):checked_scoring_scope(rows(512,'development'),'abc',manifest)
    with pytest.raises(ValueError):checked_scoring_scope(list(reversed(prompts)),'abc',manifest)
