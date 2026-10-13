from dataclasses import replace
import pytest
from posttrain.common import digest
from posttrain.data.selective_repair import repair_problem, historical_subset_evidence


def test_selective_removal_preserves_source_and_clears_validation(problem):
    original=replace(problem,validation_status='reference_passed')
    cleaned,lineage=repair_problem(original,[1],minimum_tests=3)
    assert len(original.tests)==5
    assert cleaned.tests==tuple(t for i,t in enumerate(original.tests) if i!=1)
    assert cleaned.references==original.references
    assert cleaned.validation_status=='cleaned_pending_validation'
    assert cleaned.tests_hash!=original.tests_hash
    assert lineage['retained_indices']==[0,2,3,4]
    assert lineage['original_tests_hash']==original.tests_hash


@pytest.mark.parametrize('indices',[[True],[-1],[5],[1,1],['1']])
def test_bad_removal_indices_rejected(problem,indices):
    with pytest.raises(ValueError):repair_problem(problem,indices)


def test_insufficient_tests_preserves_lineage_without_building_empty_problem(problem):
    cleaned,lineage=repair_problem(problem,list(range(5)))
    assert cleaned is None
    assert lineage['reason']=='insufficient_retained_tests'
    assert lineage['retained_indices']==[]


def evidence(problem):
    return {'problem_id':problem.problem_id,'complete':True,'audit':[{'reference_index':0,'result':{
        'tests_hash':problem.tests_hash,'code_hash':digest(problem.references[0]),
        'valid':True,'evidence_level':'isolated_execution','total':len(problem.tests),
        'cases':[{'status':'ok','exit_code':0,'truncated':False,'stdout':t.stdout} for t in problem.tests]}}]}


def test_subset_proof_is_historical_not_new_execution(problem):
    row=evidence(problem);row['audit'][0]['result']['cases'][1]['stdout']='wrong'
    assert historical_subset_evidence(problem,[0,2,3,4],row)['retained_all_pass']
    assert not historical_subset_evidence(problem,list(range(5)),row)['retained_all_pass']
    row['audit'][0]['result']['tests_hash']='changed'
    with pytest.raises(ValueError,match='identity'):
        historical_subset_evidence(problem,[0],row)


def test_bad_code_or_truncated_output_cannot_certify(problem):
    row=evidence(problem);row['audit'][0]['result']['code_hash']='wrong'
    with pytest.raises(ValueError,match='identity'):historical_subset_evidence(problem,[0],row)
    row=evidence(problem);row['audit'][0]['result']['cases'][0]['truncated']=True
    assert not historical_subset_evidence(problem,[0],row)['retained_all_pass']
