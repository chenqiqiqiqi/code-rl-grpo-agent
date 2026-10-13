import pytest
from posttrain.data.review_contracts import check_input

@pytest.mark.parametrize(('pid','valid','invalid'),[
 ('000181','3','2'),('003355','100','-101'),('005992','6','0'),
 ('002585','1 1 2','1 3 2'),('003121','1 4 1 5 0','1 4 1 1000000001 0'),
 ('005860','3 1 2 2 3','3 1 2 1 2'),('001870','1 2 1 3','1 2 1 0'),
 ('008466',' '.join(['0']*26)+' abc',' '.join(['0']*26)+' A'),
 ('009984','3 1 2 4','3 1 2 10'),('008845','1 1 1 1 1 1 1 1','2 2 1 11 11 1 1 2 2'),
 ('002803','2 1 2 2 1','2 1 1 2 1'),('004138','1 2 1 2','1 2 1 3'),
])
def test_static_domain_contracts_reject_invalid_and_trailing(pid,valid,invalid):
    assert check_input('taco-train-'+pid,valid) is None
    assert check_input('taco-train-'+pid,invalid) is not None
    assert check_input('taco-train-'+pid,valid+' extra') is not None

def test_unknown_domain_is_never_reported_valid():
    with pytest.raises(ValueError,match='unsupported_review_contract'):
        check_input('other','1')
