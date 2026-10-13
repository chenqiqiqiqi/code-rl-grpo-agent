import importlib.util
import json
from pathlib import Path
import pytest
from posttrain.common import file_digest, write_json, write_jsonl

spec = importlib.util.spec_from_file_location('completion_bundle', Path(__file__).resolve().parents[1] / 'scripts/package_stage1_completion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path):
    root = tmp_path / 'cleaning'; root.mkdir()
    write_jsonl(root / 'cleaned-problems.jsonl', [{'problem_id': f'p{i}', 'cluster_id': f'c{i//2}', 'difficulty': 'EASY'} for i in range(210)])
    report = {'complete': True, 'artifacts': {'cleaned-problems.jsonl': file_digest(root / 'cleaned-problems.jsonl')},
              'counts': {'retained_problems': 210}}
    write_json(root / 'report.json', report)
    write_json(root / 'status.json', {'complete': True, 'report_sha256': file_digest(root / 'report.json')})
    return root


def test_repaired_source_change_and_incomplete_run_refused(tmp_path):
    root = source(tmp_path)
    assert module.checked_cleaned_source(root).name == 'cleaned-problems.jsonl'
    with (root / 'cleaned-problems.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='artifact_hash_mismatch'):
        module.checked_cleaned_source(root)
    write_json(root / 'status.json', {'complete': False})
    with pytest.raises(ValueError, match='incomplete'):
        module.checked_cleaned_source(root)


def test_smoke_selection_is_fixed_and_cluster_unique(tmp_path):
    root = source(tmp_path)
    rows = list(module.smoke_ids(root / 'cleaned-problems.jsonl'))
    assert len(rows) == len(set(rows)) == 100
    assert len({int(x[1:]) // 2 for x in rows}) == 100
    assert rows == list(module.smoke_ids(root / 'cleaned-problems.jsonl'))
