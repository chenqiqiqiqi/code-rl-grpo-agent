from dataclasses import replace
import pytest

from posttrain.common import digest
from posttrain.execution.contracts import CaseResult,JudgeResult,Status
from posttrain.training.candidate_scoring import summarize_scores


def record(pid,index,passed=1,valid=True):
    r=JudgeResult(f'{pid}-{index}',Status.OK if passed else Status.WRONG,passed,1,
        (CaseResult(Status.OK if passed else Status.WRONG),),.1,0,1,0,False,'executor',
        'isolated_execution',digest(str(index)),'tests')
    if not valid:r=replace(r,status=Status.INFRA,cases=(),passed=0)
    return {'problem_id':pid,'candidate_index':index,'result':r.to_dict(),'code_hash':r.code_hash,
            'output_tokens':4,'format_rejected':False}


def test_calibration_reports_invalid_and_signal_denominators_separately():
    prompts=[{'problem_id':'a','difficulty':'easy'},{'problem_id':'b','difficulty':'hard'}]
    rows=[record('a',0,0),record('a',1,1),record('b',0),record('b',1,valid=False)]
    report=summarize_scores(prompts,rows,2)
    assert report['valid_group_fraction']==.5
    assert report['binary']['signal_group_fraction']==1.
    assert report['by_difficulty']['hard']['binary']['signal_group_fraction'] is None
    assert not report['diagnostic_thresholds_met']
    with pytest.raises(ValueError):summarize_scores(prompts,rows+[rows[0]],2)
    with pytest.raises(ValueError):summarize_scores(prompts,rows[:-1],2)


def test_single_sample_is_not_mislabeled_greedy():
    prompts=[{'problem_id':'a','difficulty':'easy'}]
    stochastic=summarize_scores(prompts,[record('a',0)],1,temperature=.8)
    assert 'greedy_pass1' not in stochastic and stochastic['sampled_pass1']==1
    greedy=summarize_scores(prompts,[record('a',0)],1,temperature=0)
    assert greedy['greedy_pass1']==1
    with pytest.raises(ValueError):summarize_scores(prompts,[record('a',0)],1)
