import copy
import pytest
from posttrain.common import digest
from posttrain.training.sft_admission import checked_admitted_rows
from posttrain.training.sft import checked_config

def fixture():
    row={'schema':'public_sft_v1','problem_id':'x','cluster_id':'cx','messages':[{'role':'system','content':'s'},{'role':'user','content':'u'}],'answer':'print(1)','source_row_sha256':'a'*64}
    entry={'problem_id':'x','cluster_id':'cx','split':'train','role':'S_gold','row_sha256':digest(row),'receipt_sha256':'b'*64,'whole_pass':True,'executor_version':'sandboxfusion-docker:test','limits':{'time_s':2.0,'memory_mb':256}}
    dev={'schema':'heldout_clusters_v1','clusters':['cy']}
    manifest={'schema':'sft_gold_admission_v1','entries':[entry],'heldout_sha256':digest(dev)}
    return [row],manifest,dev

def test_valid_admission():
    rows,m,d=fixture();assert checked_admitted_rows(rows,m,d)[0].problem_id=='x'

@pytest.mark.parametrize('change',['missing','A','dev','overlap','row'])
def test_reject_unadmitted_wrong_route_or_split(change):
    rows,m,d=fixture()
    if change=='missing':m=None
    if change=='A':m['entries'][0]['role']='A_soft'
    if change=='dev':m['entries'][0]['split']='development'
    if change=='overlap':d['clusters']=['cx'];m['heldout_sha256']=digest(d)
    if change=='row':rows[0]['answer']='print(2)'
    with pytest.raises(ValueError):checked_admitted_rows(rows,m,d)

def test_ce_unknown_objective_rejected():
    cfg={'objective':'ce_frozen_m0_forward_kl_v1'}
    with pytest.raises(ValueError,match='unknown'):checked_config(cfg)
