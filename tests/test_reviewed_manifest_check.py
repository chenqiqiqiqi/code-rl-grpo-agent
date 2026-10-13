import json
import pytest
from posttrain.common import digest, file_digest
from scripts.check_reviewed_manifest import check


def files(tmp_path, *, invalid_sample=False):
    statement = 'Input integer n.\n-----Sample Input-----\n' + ('0' if invalid_sample else '2') + '\n-----Sample Output-----\n1\n'
    evidence = [{'start': 0, 'end': 16, 'text': statement[:16]}]
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'problem_id': 'p1', 'statement': statement, 'public_examples': []}) + '\n', encoding='utf-8')
    contract = {'schema_version': 1, 'coverage': 'complete', 'sound_for_deletion': True,
                'operations': [{'id': 'head', 'kind': 'fields', 'evidence': evidence,
                                'fields': [{'kind': 'int', 'name': 'n', 'evidence': evidence,
                                            'bounds': [{'min_expr': {'constant': 1, 'terms': {}}, 'evidence': evidence}]}]}]}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_sha256': file_digest(source),
                                   'contracts_by_statement_sha256': {digest(statement): {'review_basis': 'test', 'contract': contract}}}), encoding='utf-8')
    return source, manifest


def test_public_example_conflict_fails_check(tmp_path):
    source, manifest = files(tmp_path, invalid_sample=True)
    result = check(source, manifest)
    assert not result['all_structural_and_sample_checks_pass']
    assert result['records'][0]['sample_conflicts']


def test_source_identity_change_fails_closed(tmp_path):
    source, manifest = files(tmp_path)
    source.write_text(source.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source_hash'):
        check(source, manifest)


def test_complete_contract_with_public_example_is_checked(tmp_path):
    source, manifest = files(tmp_path)
    result = check(source, manifest)
    assert result['all_structural_and_sample_checks_pass']
    assert result['records'][0]['samples_checked'] == 1
    assert result['semantic_correctness_proven'] is False
