"""Case-level cleaning preserves uncertainty and the source corpus."""
from posttrain.common import digest
from posttrain.data.domain_audit import check_contract
from posttrain.data.review_contracts import check_input
from posttrain.data.selective_repair import repair_problem
from posttrain.data.example_guard import statement_example_inputs
from posttrain.data.checker_compatibility import checker_review_flags


def _verify_evidence(evidence, statement):
    items=evidence if isinstance(evidence,list) else [evidence]
    if not items:raise ValueError('empty_rule_evidence')
    for item in items:
        if not isinstance(item,dict):raise ValueError('malformed_rule_evidence')
        start=item.get('start');end=item.get('end');text=item.get('text')
        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(statement)
                or not isinstance(text,str) or statement[start:end]!=text):
            raise ValueError('rule_evidence_not_bound_to_statement')


def clean_problem(problem, registry, compile_contract, validate_input, assess_statement, minimum_tests=5):
    statement=assess_statement(problem.statement)
    for issue in statement['issues']:
        _verify_evidence({'text':issue.get('evidence'),'start':issue.get('start'),
                          'end':issue.get('end')},problem.statement)
    rule=registry.get(problem.problem_id)
    audit={'problem_id':problem.problem_id,'cluster_id':problem.cluster_id,
        'statement_sha256':digest(problem.statement),'statement_assessment':statement,
        'checker_review_flags':checker_review_flags(problem.statement),
        'training_ready':False,'removed_indices':[],'violations':[]}
    if rule and rule['statement_sha256']!=digest(problem.statement):raise ValueError('registered_statement_changed')
    if statement['status']=='quarantine' or (rule and rule['kind']=='statement_quarantine'):
        audit.update(disposition='statement_quarantine',input_coverage='not_checked',
            registered_reason=rule.get('reason') if rule else None,
            original_tests_hash=problem.tests_hash,original_test_count=len(problem.tests))
        return None,audit
    contract=rule if rule else compile_contract(problem.statement)
    audit['contract']=contract
    for hold in contract.get('output_review_holds',[]):
        if not isinstance(hold,dict) or not isinstance(hold.get('reason'),str) or not hold['reason']:
            raise ValueError('malformed_output_review_hold')
        _verify_evidence(hold.get('evidence'),problem.statement)
        audit['checker_review_flags'].append(hold)
    if not rule and not (contract.get('sound_for_deletion') is True and contract.get('release_review_sha256')):
        cleaned,lineage=repair_problem(problem,[],minimum_tests)
        audit.update(lineage,disposition='pending_domain_verification',input_coverage='unsupported',
                     public_samples_checked=0,unreviewed_compiler_deletion_blocked=True)
        if cleaned is None:audit['disposition']='insufficient_retained_tests'
        elif audit['checker_review_flags']:audit['disposition']='pending_checker_compatibility'
        elif statement['status']=='review_needed':audit['disposition']='pending_statement_review'
        return cleaned,audit
    samples=statement_example_inputs(problem.statement)+[t.stdin for t in problem.public_examples]
    conflicts=[]
    for i,text in enumerate(samples):
        if rule:
            error=(check_input(problem.problem_id,text) if rule['kind']=='legacy_review'
                   else check_contract(rule,text))
            sample_violations=[{'reason':error}] if error else []
        else:sample_violations=validate_input(contract,text)['violations']
        if sample_violations:conflicts.append({'sample_index':i,'stdin_sha256':digest(text),'violations':sample_violations})
    audit['public_samples_checked']=len(samples)
    if conflicts:
        cleaned,lineage=repair_problem(problem,[],minimum_tests)
        audit.update(lineage,disposition='pending_rule_example_conflict',input_coverage='unsupported',
                     public_sample_conflicts=conflicts)
        return cleaned,audit
    coverage='complete' if rule else contract.get('coverage','unsupported')
    source_coverage=coverage
    for i,test in enumerate(problem.tests):
        if rule:
            error=(check_input(problem.problem_id,test.stdin) if rule['kind']=='legacy_review'
                   else check_contract(rule,test.stdin))
            violations=[{'reason':error,'evidence':{'registered_contract_sha256':digest(rule),
                        'statement_sha256':digest(problem.statement)}}] if error else []
        else:
            outcome=validate_input(contract,test.stdin)
            violations=outcome['violations']
            current=outcome.get('coverage','unsupported')
            source_coverage=min((source_coverage,current),key=lambda x:{'unsupported':0,'partial':1,'complete':2}[x])
            # Rejected cases keep their original diagnostics but are not part
            # of the retained input domain. Unknown retained cases still block.
            if not violations:
                coverage=min((coverage,current),key=lambda x:{'unsupported':0,'partial':1,'complete':2}[x])
            if any(not v.get('evidence') for v in violations):raise ValueError('violation_missing_evidence')
            for v in violations:_verify_evidence(v['evidence'],problem.statement)
        if violations:
            audit['removed_indices'].append(i)
            audit['violations'].append({'test_index':i,'stdin_sha256':digest(test.stdin),'violations':violations})
    cleaned,lineage=repair_problem(problem,audit['removed_indices'],minimum_tests)
    audit.update(lineage,input_coverage=coverage,source_input_coverage=source_coverage)
    if cleaned is None:audit['disposition']='insufficient_retained_tests'
    elif audit['checker_review_flags']:audit['disposition']='pending_checker_compatibility'
    elif statement['status']=='review_needed':audit['disposition']='pending_statement_review'
    elif coverage!='complete':audit['disposition']='pending_domain_verification'
    else:audit['disposition']='pending_reference_and_coverage_validation'
    return cleaned,audit
