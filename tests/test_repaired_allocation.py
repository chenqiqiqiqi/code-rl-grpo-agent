"""Synthetic allocation tests; these are not real reference-run evidence."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from posttrain.data.schema import Problem, TestCase as Case
from posttrain.data.repaired_allocation import allocate_verified_problems
from posttrain.common import file_digest, write_json, write_jsonl


def pool(n):
    return [Problem(str(i), 'synthetic', 'v1', str(i), 'Print the input.',
                    tuple(Case(str(j), str(j)) for j in range(5)), ('print(input())',),
                    difficulty='easy' if i % 2 else 'hard', cluster_id='cluster-'+str(i),
                    validation_status='reference_passed') for i in range(n)]


def test_protocol_capacity_and_deterministic_cluster_split():
    rows = pool(2400)
    rows[1] = replace(rows[1], cluster_id=rows[0].cluster_id)
    a = allocate_verified_problems(rows)
    b = allocate_verified_problems(list(reversed(rows)))
    assert a == b
    assert all(a['protocol_capacity_checks'].values())
    assert len(a['calibration_ids']) == 200
    assert set(a['calibration_ids']) <= set(a['train_ids'])
    assert not set(a['train_ids']) & set(a['dev_ids'])
    assert (rows[0].problem_id in a['train_ids']) == (rows[1].problem_id in a['train_ids'])
    assert a['training_ready'] is False
    assert a['boundary_sample_review_complete'] is False


def test_small_pool_records_shortfall_without_reducing_requested_calibration():
    result = allocate_verified_problems(pool(120))
    assert result['calibration_requested'] == 200
    assert result['calibration_ids'] == []
    assert not result['protocol_capacity_checks']['calibration_200_available_in_train']
    assert not result['protocol_capacity_checks']['rl_capacity_at_least_2000']
    assert result['training_ready'] is False


def test_unverified_duplicate_or_unclustered_rows_fail_closed():
    rows = pool(2)
    for bad in ([rows[0], rows[0]], [replace(rows[0], validation_status='pending')],
                [replace(rows[0], cluster_id='')],
                [replace(rows[0], references=('print(input())', 'print(0)'))]):
        with pytest.raises(ValueError):
            allocate_verified_problems(bad)


def test_single_cluster_is_an_explicit_split_blocker():
    result = allocate_verified_problems([replace(r, cluster_id='one') for r in pool(3)])
    assert result['split_complete'] is False
    assert result['train_ids'] == result['dev_ids'] == []
    assert result['split_blocker'] == 'need_at_least_two_clusters'


def cli_module():
    spec = importlib.util.spec_from_file_location('allocate_requalified_stage1',
        Path(__file__).resolve().parents[1] / 'scripts/allocate_requalified_stage1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_bundle(tmp_path):
    qualified = tmp_path / 'qualified'; qualified.mkdir()
    original = tmp_path / 'synthetic-original.json'
    write_json(original, {'synthetic': True})
    write_jsonl(qualified / 'eligible-cpu.jsonl', (p.to_dict() for p in pool(120)))
    write_json(qualified / 'source-lineage.json', {
        'source_sha256': file_digest(original), 'complete': True,
        'all_artifact_hashes_verified': True,
        'all_original_and_retained_test_hashes_verified': True,
        'original_statements_answers_references_preserved': True})
    cleaned = tmp_path / 'cleaned-problems.jsonl'
    write_jsonl(cleaned, (p.to_dict() for p in pool(120)))
    write_json(qualified / 'fresh-backup-verification.json', {
        'synthetic': True, 'scope': 'local_record_reconstruction_not_model_evaluation',
        'all_checks_pass': True, 'original_evidence_unchanged': True,
        'unknown_physical_attempts': 0, 'input_problems': 120,
        'reference_passed': 120, 'quarantined': 0, 'source_sha256': file_digest(cleaned)})
    write_json(qualified / 'report.json', {
        'scope': 'fresh_reference_and_input_domain_join_only',
        'eligible_ids': [str(i) for i in range(120)],
        'retained_source_problems': 120, 'fresh_reference_passed': 120,
        'implementation_sha256': {str(p): file_digest(p) for p in (
            Path('scripts/requalify_repaired_pool.py'), Path('src/posttrain/data/requalification.py'))},
        'input_files_sha256': {str(original): file_digest(original), str(cleaned): file_digest(cleaned)},
        'artifacts': {p.name:file_digest(p) for p in qualified.iterdir()}})
    final = tmp_path / 'final'; final.mkdir()
    for name in ('humaneval_plus.json', 'lcb_stdio_200.json', 'mbpp_plus.json'):
        write_json(final / name, {'synthetic': True})
    dedup = tmp_path / 'dedup.json'
    write_json(dedup, {'candidate_manifest_sha256': file_digest(original),
        'removed_final_overlap_ids': [], 'hidden_final_tests_loaded': False,
        'final_manifests': {p.name:file_digest(p) for p in final.iterdir()}})
    policy = tmp_path / 'policy.json'
    write_json(policy, json.loads(Path('manifests/review_policy_20260927.json').read_text(encoding='utf-8')))
    return qualified, file_digest(qualified / 'report.json'), dedup, final, policy


def test_cli_creates_only_provisional_bound_outputs(tmp_path):
    args = synthetic_bundle(tmp_path)
    output = tmp_path / 'allocation'
    result = cli_module().allocate(*args, output)
    assert result['training_ready'] is False
    assert result['boundary_sample_review_complete'] is False
    assert result['split_complete'] is True
    prompts = [json.loads(x) for x in (output / 'train-prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert prompts and all('tests' not in p and 'references' not in p for p in prompts)
    with pytest.raises(FileExistsError):
        cli_module().allocate(*args, output)


@pytest.mark.parametrize('mutation', ['eligible', 'final_public', 'report_pin'])
def test_cli_rejects_evidence_mutation_before_output(tmp_path, mutation):
    args = list(synthetic_bundle(tmp_path))
    if mutation == 'eligible':
        (args[0] / 'eligible-cpu.jsonl').write_text('{}\n')
    elif mutation == 'final_public':
        (args[3] / 'lcb_stdio_200.json').write_text('{}')
    else:
        args[1] = '0' * 64
    output = tmp_path / 'allocation'
    with pytest.raises(ValueError):
        cli_module().allocate(*args, output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['backup_failed', 'unknown_attempts', 'backup_count',
    'backup_source', 'implementation_changed', 'empty_final_manifests', 'automatic_admission'])
def test_cli_rejects_semantically_incomplete_evidence_even_when_rehashed(tmp_path, mutation):
    args = list(synthetic_bundle(tmp_path))
    qualified = args[0]
    report_path = qualified / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    backup_path = qualified / 'fresh-backup-verification.json'
    backup = json.loads(backup_path.read_text(encoding='utf-8'))
    if mutation == 'backup_failed':
        backup['all_checks_pass'] = False
    elif mutation == 'unknown_attempts':
        backup['unknown_physical_attempts'] = 1
    elif mutation == 'backup_count':
        backup['input_problems'] = 121
    elif mutation == 'backup_source':
        backup['source_sha256'] = '0' * 64
    elif mutation == 'implementation_changed':
        report['implementation_sha256'] = {'scripts/requalify_repaired_pool.py': '0' * 64}
    elif mutation == 'empty_final_manifests':
        d = json.loads(args[2].read_text(encoding='utf-8')); d['final_manifests'] = {}; write_json(args[2], d)
    elif mutation == 'automatic_admission':
        policy = json.loads(args[4].read_text(encoding='utf-8')); policy['automatic_admission_after_waiver'] = True
        write_json(args[4], policy)
    write_json(backup_path, backup)
    report['artifacts']['fresh-backup-verification.json'] = file_digest(backup_path)
    write_json(report_path, report)
    args[1] = file_digest(report_path)
    output = tmp_path / 'allocation'
    with pytest.raises(ValueError):
        cli_module().allocate(*args, output)
    assert not output.exists()
