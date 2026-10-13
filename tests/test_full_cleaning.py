from dataclasses import replace
from posttrain.common import digest
from posttrain.data.full_cleaning import clean_problem


def assess(statement):return {'status':'no_confirmed_issue','issues':[]}
def compile_empty(statement):return {'coverage':'unsupported'}
def validate_empty(contract,text):return {'coverage':'unsupported','violations':[]}
def trusted(coverage):return {'coverage':coverage,'sound_for_deletion':True,'release_review_sha256':'test-fixture'}


def test_partial_rules_do_not_promote_unknown_inputs(problem):
    p,a=clean_problem(problem,{},compile_empty,validate_empty,assess,minimum_tests=3)
    assert len(p.tests)==5
    assert a['input_coverage']=='unsupported'
    assert a['disposition']=='pending_domain_verification'
    assert not a['training_ready']


def test_known_bad_test_is_removed_without_dropping_problem(problem):
    registry={problem.problem_id:{'kind':'integer','min':0,'max':3,'statement_sha256':digest(problem.statement)}}
    p,a=clean_problem(problem,registry,compile_empty,validate_empty,assess,minimum_tests=3)
    assert len(p.tests)==4
    assert a['removed_indices']==[4]
    assert a['input_coverage']=='complete'
    assert a['disposition']=='pending_reference_and_coverage_validation'
    assert len(problem.tests)==5


def test_missing_statement_overrides_input_validity(problem):
    p,a=clean_problem(problem,{},compile_empty,validate_empty,
        lambda s:{'status':'quarantine','issues':[{'reason':'missing task','evidence':s,'start':0,'end':len(s)}]},minimum_tests=3)
    assert p is None
    assert a['disposition']=='statement_quarantine'
    assert a['removed_indices']==[]


def test_compiler_only_deletes_evidenced_violations(problem):
    problem=replace(problem,statement='n <= 3')
    def validate(c,t):return {'coverage':'partial','violations':[
        {'reason':'bound','evidence':{'text':'n <= 3','start':0,'end':6}}] if t.startswith('4') else []}
    p,a=clean_problem(problem,{},lambda s:trusted('partial'),validate,assess,minimum_tests=3)
    assert len(p.tests)==4
    assert a['input_coverage']=='partial'
    assert a['disposition']=='pending_domain_verification'


def test_false_evidence_span_aborts_instead_of_deleting(problem):
    import pytest
    def validate(c,t):return {'coverage':'partial','violations':[
        {'reason':'bound','evidence':{'text':'n <= 3','start':0,'end':6}}]}
    with pytest.raises(ValueError,match='evidence'):
        clean_problem(problem,{},lambda s:trusted('partial'),validate,assess)


def test_rejecting_public_sample_disables_deletion_and_marks_conflict(problem):
    problem=replace(problem,statement='n <= 3\nExamples\nInput\n4\nOutput\n5\n')
    def validate(c,t):return {'coverage':'complete','violations':[
        {'reason':'bound','evidence':{'text':'n <= 3','start':0,'end':6}}] if t.startswith('4') else []}
    cleaned,audit=clean_problem(problem,{},lambda s:trusted('complete'),validate,assess)
    assert len(cleaned.tests)==5
    assert audit['removed_indices']==[]
    assert audit['disposition']=='pending_rule_example_conflict'


def test_removed_invalid_case_downgrade_remains_in_source_not_retained_coverage(problem):
    problem=replace(problem,statement='n <= 3')
    def validate(c,t):
        if t.startswith('4'):
            return {'coverage':'partial','violations':[{'reason':'bound','evidence':{'text':'n <= 3','start':0,'end':6}}]}
        return {'coverage':'complete','violations':[]}
    cleaned,audit=clean_problem(problem,{},lambda s:trusted('complete'),validate,assess,minimum_tests=3)
    assert len(cleaned.tests)==4
    assert audit['source_input_coverage']=='partial'
    assert audit['input_coverage']=='complete'


def test_generic_complete_compiler_cannot_authorize_test_deletion(problem):
    def invalid_validator(c,t):raise AssertionError('unreviewed rules must not be applied')
    cleaned,audit=clean_problem(problem,{},lambda s:{'coverage':'complete','sound_for_deletion':True},
                               invalid_validator,assess)
    assert len(cleaned.tests)==len(problem.tests)
    assert audit['input_coverage']=='unsupported'
    assert audit['unreviewed_compiler_deletion_blocked']


def test_reviewed_output_hold_blocks_admission_without_deleting_inputs(problem):
    evidence={'start':0,'end':len(problem.statement),'text':problem.statement}
    c=trusted('complete') | {'output_review_holds':[{'reason':'manual_non_unique_output','evidence':[evidence]}]}
    cleaned,audit=clean_problem(problem,{},lambda s:c,
        lambda c,t:{'coverage':'complete','violations':[]},assess)
    assert len(cleaned.tests)==len(problem.tests)
    assert audit['disposition']=='pending_checker_compatibility'
    assert audit['checker_review_flags'][0]['reason']=='manual_non_unique_output'


def test_unknown_retained_case_still_blocks_complete_coverage(problem):
    def validate(c,t):
        return {'coverage':'partial' if t.startswith('4') else 'complete','violations':[]}
    cleaned,audit=clean_problem(problem,{},lambda s:trusted('complete'),validate,assess)
    assert len(cleaned.tests)==5
    assert audit['input_coverage']=='partial'
    assert audit['source_input_coverage']=='partial'
    assert audit['disposition']=='pending_domain_verification'
