from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bind_reviewed_cleaning_successor import bind_pairs


def row(indices):
    return {'p': {'statement_sha256': 's', 'original_tests_hash': 't',
                  'parent_record_sha256': 'r', 'removed_indices': indices}}


def test_restoration_transfers_only_supported_remaining_deletions():
    new, restored = bind_pairs(row([0]), row([0, 1, 2]), row([0, 2]), {('p', 1), ('p', 2)})
    assert new == {('p', 2)} and restored == {('p', 1)}


def test_unsupported_or_changed_source_fails_closed():
    with pytest.raises(ValueError, match='unreviewed_deletion'):
        bind_pairs(row([0]), row([0, 1]), row([0, 2]), {('p', 2)})
    with pytest.raises(ValueError, match='missing_independent_support'):
        bind_pairs(row([0]), row([0, 1]), row([0, 1]), set())
    changed = row([0]); changed['p']['original_tests_hash'] = 'different'
    with pytest.raises(ValueError, match='source_identity_changed'):
        bind_pairs(row([0]), row([0]), changed, set())
