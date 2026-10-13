"""Join an already verified fresh reference run to its exact cleaned source.

The caller must first verify the run archive and bind its input SHA256 to the
cleaned pool. This pure join neither authenticates a run nor authorizes training.
Historical subset results do not determine eligibility after a fresh run.
"""
from dataclasses import replace

from posttrain.common import digest
from posttrain.data.schema import Problem


def _unique(rows, label):
    result = {}
    for row in rows:
        pid = row['problem_id']
        if pid in result:
            raise ValueError('duplicate_' + label)
        result[pid] = row
    return result


def join_verified_fresh_results(source_rows, accepted_rows, audit_rows):
    sources = _unique(source_rows, 'source')
    accepted = _unique(accepted_rows, 'accepted')
    audits = _unique(audit_rows, 'audit')
    if not accepted.keys() <= sources.keys() or not sources.keys() <= audits.keys():
        raise ValueError('fresh_source_or_audit_coverage_mismatch')
    eligible, pending = [], []
    for pid, row in sorted(sources.items()):
        source = Problem.from_dict(row)
        if digest(source.to_dict()) != digest(row):
            raise ValueError('source_not_canonical_or_contains_duplicate_tests')
        audit = audits[pid]
        if (audit['statement_sha256'] != digest(source.statement)
                or audit['cleaned_tests_hash'] != source.tests_hash
                or audit['retained_test_count'] != len(source.tests)):
            raise ValueError('cleaning_audit_mismatch:' + pid)
        reasons = []
        if pid not in accepted:
            reasons.append('no_fresh_reference_pass')
        else:
            fresh = Problem.from_dict(accepted[pid])
            if len(fresh.references) != 1 or fresh.references[0] not in source.references:
                raise ValueError('accepted_reference_not_from_source:' + pid)
            expected = replace(source, references=fresh.references, validation_status='reference_passed')
            if digest(expected.to_dict()) != digest(accepted[pid]):
                raise ValueError('accepted_problem_changed:' + pid)
        if audit.get('input_coverage') != 'complete':
            reasons.append('input_domain_not_complete')
        if len(source.tests) < 5:
            reasons.append('fewer_than_five_unique_tests')
        if not source.cluster_id:
            reasons.append('missing_cluster_identity')
        if audit.get('statement_assessment', {}).get('status') != 'no_confirmed_issue':
            reasons.append('statement_review_pending')
        if audit.get('checker_review_flags'):
            reasons.append('output_checker_incompatible_or_unresolved')
        if audit.get('oracle_review_required'):
            reasons.append('oracle_conflict_unresolved')
        if audit.get('public_samples_checked', 0) < 1:
            reasons.append('public_input_regression_missing')
        if reasons:
            pending.append({'problem_id': pid, 'cluster_id': source.cluster_id,
                            'reasons': reasons})
        else:
            eligible.append(pid)
    return {'scope': 'fresh_reference_and_input_domain_join_only',
            'eligible_ids': eligible, 'pending': pending,
            'retained_source_problems': len(sources), 'fresh_reference_passed': len(accepted),
            'training_ready': False, 'boundary_sample_review_complete': False,
            'split_complete': False, 'archive_verification_performed_by_this_function': False,
            'historical_candidate_flag_used_as_gate': False}
