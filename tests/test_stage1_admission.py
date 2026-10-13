from dataclasses import replace
import pytest
from posttrain.data.admission import admission_split, verify_bound_files


def samples(problem):
    return [replace(problem,problem_id=f'p{i}',cluster_id=f'c{i//2}',
                    validation_status='reference_passed') for i in range(20)]


def review(ids):
    return {'source_accepted_sha256':'bound','human_review_required':False,
            'quarantine_ids':['p0'],'domain_valid_ids':ids}


def test_quarantine_covers_cluster_and_calibration_is_train_only(problem):
    result=admission_split(samples(problem),review([f'p{i}' for i in range(2,20)]),'bound',3)
    assert set(result['quarantined_ids'])=={'p0','p1'}
    assert not set(result['train_ids']) & set(result['dev_ids'])
    assert not {result['cluster_by_id'][i] for i in result['train_ids']} & {
        result['cluster_by_id'][i] for i in result['dev_ids']}
    assert set(result['calibration_ids']) <= set(result['train_ids'])
    assert result['SFT_RL_overlap']==len(result['train_ids'])
    assert result['data_domain_gate_passed']
    assert not result['human_review_complete']
    assert result==admission_split(list(reversed(samples(problem))),review([f'p{i}' for i in range(2,20)]),'bound',3)


def test_unchecked_domains_block_admission_even_when_human_requirement_waived(problem):
    result=admission_split(samples(problem),review(['p2']),'bound',3)
    assert result['domain_unchecked_count']==17
    assert not result['data_domain_gate_passed']
    assert not result['training_ready']


def test_review_must_bind_current_source_and_known_ids(problem):
    with pytest.raises(ValueError,match='review_source'):
        admission_split(samples(problem),review([]),'changed',3)
    with pytest.raises(ValueError,match='review_ids'):
        admission_split(samples(problem),review(['missing']),'bound',3)
    with pytest.raises(ValueError,match='review_ids'):
        admission_split(samples(problem),review(['p0']),'bound',3)


def test_unvalidated_references_and_duplicate_ids_rejected(problem):
    ps=samples(problem)
    with pytest.raises(ValueError,match='reference'):
        admission_split([replace(ps[0],validation_status='unverified')]+ps[1:],review([]),'bound',3)
    with pytest.raises(ValueError,match='unique'):
        admission_split(ps+[ps[0]],review([]),'bound',3)


def test_evidence_hashes_and_paths_fail_closed(tmp_path):
    from posttrain.common import file_digest
    p=tmp_path/'record.json';p.write_text('original')
    hashes={'record.json':file_digest(p)}
    verify_bound_files(tmp_path,hashes)
    p.write_text('changed')
    with pytest.raises(ValueError,match='artifact_hash'):
        verify_bound_files(tmp_path,hashes)
    with pytest.raises(ValueError,match='artifact_path'):
        verify_bound_files(tmp_path,{'../outside': 'anything'})
