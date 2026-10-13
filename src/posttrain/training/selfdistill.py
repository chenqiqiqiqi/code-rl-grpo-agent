"""Strict training-target admission, separate from development model selection."""
from posttrain.common import digest
from posttrain.evaluation.extraction import extract_python

def admission_reason(candidate,record,prompt):
    if record['state']!='complete':raise ValueError('unreconciled_training_judgment')
    if (candidate['request_id']!=record['request_id'] or candidate['problem_id']!=record['problem_id'] or
        candidate['problem_id']!=prompt['problem_id'] or candidate['output_tokens']!=record['output_tokens']):
        raise ValueError('training_judgment_identity_changed')
    result=record['result'];code=extract_python(candidate['text'])
    if (record['format_rejected']!=(code is None) or
        record['code_hash']!=digest(code if code is not None else 'def <POSTTRAIN_FORMAT_REJECTED>(\n') or
        result['evidence_level']!='isolated_execution' or result['binary_valid'] is not True or
        result['tests_hash']!=prompt['tests_hash'] or result['total']!=prompt['test_count']):
        raise ValueError('training_evidence_mismatch')
    whole=(result['passed']==result['total'])
    if bool(record['binary_success'])!=whole:raise ValueError('binary_label_mismatch')
    if not whole:return 'not_whole_pass'
    if len(result['cases'])!=result['total'] or any(c['status']!='ok' for c in result['cases']):
        raise ValueError('all_cases_required_for_positive_target')
    if code is None:raise ValueError('format_rejection_cannot_pass')
    if candidate['finish_reason']!='stop' or candidate['output_tokens']>=1024:return 'not_natural_stop'
    return None
