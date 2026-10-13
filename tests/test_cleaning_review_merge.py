import pytest

from posttrain.common import digest, file_digest, write_json
from scripts.merge_cleaning_review_batches import assemble


def test_review_merge_rejects_unbound_replacement(tmp_path):
    base, upgrade = tmp_path / 'base.json', tmp_path / 'upgrade.json'
    old = {'contract': {'coverage': 'partial'}}
    write_json(base, {'source_sha256': 'same-source',
                     'contracts_by_statement_sha256': {'statement': old}})
    proposed = {'source_sha256': 'same-source',
                'base_release_manifest_sha256': file_digest(base),
                'contracts_by_statement_sha256': {'statement': {
                    'old_entry_sha256': 'stale-review',
                    'contract': {'coverage': 'complete'}}}}
    write_json(upgrade, proposed)
    with pytest.raises(ValueError, match='upgrade_entry_mismatch'):
        assemble(base, [], [upgrade])
    proposed['contracts_by_statement_sha256']['statement']['old_entry_sha256'] = digest(old)
    write_json(upgrade, proposed)
    merged = assemble(base, [], [upgrade])
    assert merged['entry_changes'][0]['before_entry_sha256'] == digest(old)
    assert merged['contracts_by_statement_sha256']['statement']['contract']['coverage'] == 'complete'


def test_new_batch_cannot_silently_overwrite_existing_review(tmp_path):
    base, addition = tmp_path / 'base.json', tmp_path / 'addition.json'
    write_json(base, {'source_sha256': 'same-source',
                     'contracts_by_statement_sha256': {'statement': {'contract': {'coverage': 'partial'}}}})
    write_json(addition, {'source_sha256': 'same-source',
                         'contracts_by_statement_sha256': {'statement': {'contract': {'coverage': 'complete'}}}})
    with pytest.raises(ValueError, match='overlapping_addition'):
        assemble(base, [addition], [])
