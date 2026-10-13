"""Independent structural checks of a derived cleaning artifact.

This module does not invoke the cleaner/compiler and cannot establish that an
input contract is semantically correct. It verifies immutable lineage instead.
"""
from dataclasses import asdict
from posttrain.common import digest


def verify_lineage(original,audit,cleaned):
    if audit['problem_id']!=original.problem_id:raise ValueError('audit_problem_identity')
    if audit['statement_sha256']!=digest(original.statement):raise ValueError('statement_identity')
    if audit['original_tests_hash']!=original.tests_hash:raise ValueError('original_tests_identity')
    if audit['original_test_count']!=len(original.tests):raise ValueError('original_test_count')
    removed=audit['removed_indices']
    if any(type(i) is not int for i in removed) or len(set(removed))!=len(removed):
        raise ValueError('invalid_removed_indices')
    if any(not 0<=i<len(original.tests) for i in removed):raise ValueError('removed_index_out_of_range')
    if audit['disposition']=='statement_quarantine':
        if cleaned is not None or removed:raise ValueError('statement_quarantine_changed_tests')
        return
    kept=audit['retained_indices']
    excluded=set(removed)
    if kept!=[i for i in range(len(original.tests)) if i not in excluded]:
        raise ValueError('retained_partition_mismatch')
    for label,indices in (('removed',removed),('retained',kept)):
        expected={str(i):digest(asdict(original.tests[i])) for i in indices}
        if audit[label+'_test_hashes']!=expected:raise ValueError('lineage_test_hash_mismatch')
    if audit['new_executions']!=0:raise ValueError('unexpected_execution_claim')
    if cleaned is None:
        if len(kept)>=5:raise ValueError('unexpected_full_problem_quarantine')
        return
    for name in ('problem_id','source','revision','source_id','statement','difficulty','interface',
                 'public_examples','time_limit_s','cluster_id'):
        if getattr(original,name)!=getattr(cleaned,name):raise ValueError('changed_problem_field:'+name)
    if cleaned.tests!=tuple(original.tests[i] for i in kept):raise ValueError('retained_tests_changed')
    if audit['cleaned_tests_hash']!=cleaned.tests_hash:raise ValueError('cleaned_test_hash')
    if cleaned.validation_status!='cleaned_pending_validation':raise ValueError('unproven_validation_status')
    if audit['training_ready'] is not False:raise ValueError('unproven_training_readiness')
    proof=audit['historical_reference_evidence']
    if proof['new_executions']!=0 or proof['fresh_reference_backtest'] is not False:
        raise ValueError('unproven_fresh_execution')
    if proof['retained_all_pass']:
        index=proof['selected_reference_index']
        if type(index) is not int or not 0<=index<len(original.references):raise ValueError('selected_reference_index')
        expected=(original.references[index],)
    else:expected=original.references
    if cleaned.references!=expected:raise ValueError('reference_source_changed')
