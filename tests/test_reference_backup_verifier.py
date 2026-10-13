import importlib.util
from pathlib import Path
import pytest
from posttrain.common import file_digest, write_json
from test_reference_pool import completed_fixture


def verifier():
    spec = importlib.util.spec_from_file_location('reference_backup_verifier', Path('scripts/verify_reference_backup.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_reconstructs_fixture_without_modifying_original_evidence(tmp_path, problem):
    pool, output, run = completed_fixture(tmp_path, problem)
    pool.collect(output)
    before = {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob('*') if p.is_file()}
    result = verifier().verify(output, file_digest(output/'report.json'), tmp_path/'scratch')
    assert result['all_checks_pass'] and result['input_problems'] == 1
    assert result['physical_executions'] == 5
    assert result['new_physical_executions'] == 0
    assert before == {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob('*') if p.is_file()}


def test_summary_tampering_and_wrong_report_digest_are_rejected(tmp_path, problem):
    pool, output, run = completed_fixture(tmp_path, problem)
    report = pool.collect(output)
    with pytest.raises(ValueError, match='report_hash'):
        verifier().verify(output, 'wrong', tmp_path/'scratch')
    report['physical_executions'] = 0
    write_json(output/'report.json', report)
    with pytest.raises(ValueError, match='reconstructed_report'):
        verifier().verify(output, file_digest(output/'report.json'), tmp_path/'scratch')


def test_unregistered_attempt_is_not_ignored(tmp_path, problem):
    pool, output, run = completed_fixture(tmp_path, problem)
    pool.collect(output)
    write_json(run/'records/unknown.json', {'complete': False})
    with pytest.raises(ValueError, match='unregistered'):
        verifier().verify(output, file_digest(output/'report.json'), tmp_path/'scratch')
