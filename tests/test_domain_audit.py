import pytest
from posttrain.common import digest
from posttrain.data.domain_audit import audit_problem, check_contract


def test_strict_scalar_contract():
    rule={'kind':'integer','min':1,'max':100}
    assert check_contract(rule,'100\n') is None
    for value in ['0','101','1 2','1.0','１','1_0','']:
        assert check_contract(rule,value) is not None


def test_string_constraints_include_semantic_condition():
    rule={'kind':'string','min':3,'max':5,'alphabet':'abc','adjacent_distinct':True}
    assert check_contract(rule,'abca') is None
    for value in ['aab','abz','ab','abcabc','abc x']:
        assert check_contract(rule,value) is not None


def test_unknown_never_passes():
    row={'problem_id':'unknown','statement':'task','tests':[{'stdin':'1'}]}
    assert audit_problem(row,{})['status']=='unsupported'


def test_statement_mutation_is_not_admitted():
    row={'problem_id':'x','statement':'changed','tests':[{'stdin':'1'}]}
    rule={'statement_sha256':digest('original'),'kind':'integer','min':1,'max':2}
    with pytest.raises(ValueError,match='statement_changed'):
        audit_problem(row,{'x':rule})


def test_evidence_counts_all_tests_and_keeps_failure_index():
    row={'problem_id':'x','statement':'task','tests':[{'stdin':'1'},{'stdin':'3'}]}
    rule={'statement_sha256':digest('task'),'kind':'integer','min':1,'max':2}
    result=audit_problem(row,{'x':rule})
    assert result['status']=='invalid_input'
    assert result['tests_checked']==2
    assert result['violations'][0]['test_index']==1
    assert result['violations'][0]['stdin_sha256']==digest('3')


def test_ambiguous_statement_never_becomes_domain_pass():
    row={'problem_id':'x','statement':'task','tests':[{'stdin':'1'}]}
    rule={'statement_sha256':digest('task'),'kind':'statement_quarantine','reason':'incomplete'}
    result=audit_problem(row,{'x':rule})
    assert result['status']=='statement_quarantine'
    assert result['tests_checked']==0


def test_whole_pool_audit_preserves_unknowns_and_quarantines_cluster(tmp_path):
    import json
    from posttrain.common import file_digest,write_json,write_jsonl
    from scripts.audit_input_domains import run
    source=tmp_path/'input.jsonl'; registry=tmp_path/'registry.json';out=tmp_path/'out'
    rows=[{'problem_id':key,'cluster_id':cluster,'statement':'task','tests':[{'stdin':value}]}
          for key,cluster,value in [('a','same','9'),('b','same','1'),('c','other','1')]]
    write_jsonl(source,rows)
    write_json(registry,{'source_sha256':file_digest(source),'code_dependencies':{},'contracts':{
        'a':{'statement_sha256':digest('task'),'kind':'integer','min':1,'max':2}}})
    result=run(source,registry,out)
    assert result['problem_count']==3
    assert result['statuses']=={'invalid_input':1,'unsupported':2}
    assert result['quarantined_problem_ids']==['a','b']
    assert not result['training_ready']
    assert result['artifacts']['per_problem.jsonl']==file_digest(out/'per_problem.jsonl')
    with pytest.raises(ValueError,match='immutable'):
        run(source,registry,out)
    source.write_text('{}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='source_changed'):
        run(source,registry,tmp_path/'changed')
