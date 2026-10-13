import pytest

from posttrain.evaluation.extraction import extract_python
from posttrain.evaluation.statistics import paired_report


def test_paired_bootstrap_reuses_task_indices_across_seeds():
    rows=[{'task_id':str(i),'difficulty':'easy','m2_42':False,'m3_42':True,'m2_43':True,'m3_43':False} for i in range(8)]
    report=paired_report(rows,bootstrap=100)
    assert report['mean_difference_pp']==0
    assert report['paired_difference_pp']=={'42':100,'43':-100}
    assert report['bootstrap_95_percent_pp']==[0,0]
    rows[0]['m2_42']=None
    with pytest.raises(ValueError):
        paired_report(rows)


def test_extraction_does_not_guess_multiple_code_blocks():
    assert extract_python('```python\nprint(1)\n```')=='print(1)\n'
    assert extract_python('```python\nprint(1)\n```\n```python\nprint(2)\n```') is None
    assert extract_python('def bad(') is None
