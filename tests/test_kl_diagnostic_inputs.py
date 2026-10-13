import pytest
from posttrain.training.kl_diagnostic import trajectory_sample, validate_sample

def test_sampled_ids_are_preserved_including_eos():
    s=trajectory_sample([10,11],[20,151645],2,2,151645)
    assert s['input_ids']==[10,11,20,151645]
    assert s['labels']==[-100,-100,20,151645]
    assert validate_sample(s)==2

@pytest.mark.parametrize('prefix,ids,n,m', [([10],[20,151645],2,2),([10],[20],1,1),([10],[20,151645],1,3)])
def test_raw_trajectory_identity_mismatch_fails(prefix,ids,n,m):
    with pytest.raises(ValueError):trajectory_sample(prefix,ids,n,m,151645)

def test_corrupt_causal_labels_fail():
    with pytest.raises(ValueError):validate_sample({'input_ids':[10,20], 'labels':[-100,21],'attention_mask':[1,1]})
