"""Versioned deletion of proven illegal inputs, never a fresh execution claim."""
from dataclasses import asdict, replace
from posttrain.common import digest
from posttrain.data.schema import normalize_output


def _indices(values, size):
    values=list(values)
    if any(type(i) is not int or not 0<=i<size for i in values) or len(values)!=len(set(values)):
        raise ValueError('invalid_or_duplicate_test_indices')
    return sorted(values)


def repair_problem(problem, invalid_indices, minimum_tests=5):
    if type(minimum_tests) is not int or minimum_tests<1:raise ValueError('invalid_minimum_tests')
    removed=_indices(invalid_indices,len(problem.tests));excluded=set(removed)
    kept=[i for i in range(len(problem.tests)) if i not in excluded]
    lineage={'problem_id':problem.problem_id,'original_tests_hash':problem.tests_hash,
        'removed_indices':removed,'retained_indices':kept,
        'removed_test_hashes':{str(i):digest(asdict(problem.tests[i])) for i in removed},
        'retained_test_hashes':{str(i):digest(asdict(problem.tests[i])) for i in kept},
        'new_executions':0,'original_test_count':len(problem.tests),'retained_test_count':len(kept)}
    if len(kept)<minimum_tests:
        return None,{**lineage,'reason':'insufficient_retained_tests'}
    cleaned=replace(problem,tests=tuple(problem.tests[i] for i in kept),
                    validation_status='cleaned_pending_validation',metadata={**problem.metadata,
                    'cleaning_parent_tests_hash':problem.tests_hash})
    lineage['cleaned_tests_hash']=cleaned.tests_hash
    return cleaned,lineage


def historical_subset_evidence(problem, retained_indices, record):
    kept=_indices(retained_indices,len(problem.tests))
    if record['problem_id']!=problem.problem_id or not record.get('complete'):
        raise ValueError('historical_record_identity_or_completion')
    attempts=[]
    for audit in record['audit']:
        if 'result' not in audit:continue
        index=audit['reference_index'];result=audit['result']
        if type(index) is not int or not 0<=index<len(problem.references):
            raise ValueError('historical_reference_identity')
        if result['code_hash']!=digest(problem.references[index]) or result['tests_hash']!=problem.tests_hash:
            raise ValueError('historical_code_or_tests_identity')
        if result['total']!=len(problem.tests) or len(result['cases'])!=len(problem.tests):
            raise ValueError('historical_case_coverage')
        valid=result.get('valid') is True and result.get('evidence_level')=='isolated_execution'
        passing=[]
        for i in kept:
            case=result['cases'][i]
            if (valid and case.get('status')=='ok' and case.get('exit_code')==0
                    and not case.get('truncated',False)
                    and normalize_output(case['stdout'])==normalize_output(problem.tests[i].stdout)):
                passing.append(i)
        attempts.append({'reference_index':index,'reference_sha256':result['code_hash'],
                         'retained_passed':len(passing),'retained_total':len(kept),
                         'retained_all_pass':bool(kept) and len(passing)==len(kept)})
    successful=next((r for r in attempts if r['retained_all_pass']),None)
    return {'evidence_level':'historical_isolated_case_subset_derivation','new_executions':0,
            'fresh_reference_backtest':False,'retained_all_pass':successful is not None,
            'selected_reference_index':successful['reference_index'] if successful else None,
            'references':attempts}
