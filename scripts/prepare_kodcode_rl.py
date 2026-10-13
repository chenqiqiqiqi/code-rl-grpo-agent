"""Build KodCode-V1-SFT-4o function tasks for RL on the unchanged stdio scorer (local, CPU only).

Stage `build`: filter the pinned shard (Algorithm+Evol, 4o-correct, medium/hard, plain assert tests,
stdlib only, unique test functions), 10-gram decontaminate against the development statements and the
final benchmarks (the diagnostic study's decontam-statements.jsonl), shuffle (seed 42), and write
hidden VM problem rows (one hidden test program per test function, secret token as expected stdout)
plus reference candidates for verification on the isolated executor. Nothing here goes to the GPU.

Stage `finalize`: keep problems whose 4o reference passed every hidden test on the VM, write the VM
problem file (validation_status reference_passed) and the public GPU prompts (no tests, no tokens):
`rl` pool for screening/training and a held-out `probe` set for in-distribution evaluation.
License: KodCode is CC BY-NC 4.0 (non-commercial research use).
"""
import argparse
import ast
import json
from pathlib import Path
import random
import re
import secrets
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from posttrain.common import digest, file_digest, utc_now, write_json, write_jsonl
from posttrain.data.prompt_manifest import checked_prompt
from posttrain.data.schema import Problem, TestCase
from posttrain.execution.function_harness import FUNCTION_TASK_PREFIX, SYSTEM_PROMPT, hidden_programs, test_names

SHARD = ROOT/'data/external/kodcode-v1-sft-4o/train-00000-of-00005.parquet'
SHARD_SHA256 = '3e846c70bc682296f87d680fffdc388b6464bad9a45713188c6b30e628cc3b39'
DECONTAM = ROOT/'claude-diag-20260930/decontam-statements.jsonl'
STDLIB = set(sys.stdlib_module_names)
ALPHABET = string.ascii_letters + string.digits


def grams(text, n=10):
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def seed_ids(row):
    meta = row.get('metadata') or {}
    if isinstance(meta, str):
        meta = ast.literal_eval(meta)
    seeds = meta.get('seed_ids') or []
    if isinstance(seeds, str):
        seeds = re.findall(r'[\w\-]+', seeds)
    return {str(s) for s in seeds}


def imports(code):
    return {m.split('.')[0] for m in re.findall(r'(?m)^\s*(?:from|import)\s+([\w.]+)', code)}


def statement(row):
    """The question plus any declared function the tests import but the question never names."""
    imported = {n.strip() for group in re.findall(r'from solution import ([\w, ]+)', row['test_code'])
                for n in group.split(',') if n.strip()}
    info = row['test_info'] or []
    if isinstance(info, str):
        info = ast.literal_eval(info)
    declared = {d.get('function_name'): d.get('function_declaration') for d in info if isinstance(d, dict)}
    missing = sorted(n for n in imported if n not in row['question'])
    if any(not declared.get(n) for n in missing):
        return None
    extra = ''.join(f'\n{declared[n]}' for n in missing)
    return row['question'].strip() + (f'\n\nImplement:{extra}' if extra else '')


def eligible(rows, bad):
    for r in rows:
        if r['subset'] not in ('Algorithm', 'Evol') or r['4o_correctness'] != 'True':
            continue
        if r['gpt_difficulty'] not in ('medium', 'hard') or 'pytest' in r['test_code']:
            continue
        if not imports(r['4o_solution']) <= STDLIB or not imports(r['test_code']) - {'solution'} <= STDLIB:
            continue
        try:
            names = test_names(r['test_code'])
        except ValueError:
            continue
        text = statement(r)
        if text is None or not 2 <= len(names) <= 30 or len(text) > 6000 or grams(r['question']) & bad:
            continue
        yield r, text


def build(out, n_pool, n_probe):
    if out.exists():
        raise ValueError('immutable_output_exists')
    if file_digest(SHARD) != SHARD_SHA256:
        raise ValueError('pinned_shard_changed')
    import pyarrow.parquet as pq
    bad = set()
    for line in DECONTAM.read_text(encoding='utf-8').splitlines():
        bad |= grams(json.loads(line)['text'])
    rows = pq.read_table(SHARD).to_pylist()
    cand = list(eligible(rows, bad))
    cand.sort(key=lambda x: x[0]['question_id'])
    random.Random(42).shuffle(cand)
    chosen = cand[:n_pool + n_probe]
    problems, refs = [], []
    for i, (r, text) in enumerate(chosen):
        token = ''.join(secrets.choice(ALPHABET) for _ in range(24))
        pid = FUNCTION_TASK_PREFIX + r['question_id']
        tests = tuple(TestCase(stdin=s, stdout=o) for s, o in hidden_programs(r['test_code'], token))
        p = Problem(problem_id=pid, source='kodcode-v1-sft-4o', revision='14f8782f', source_id=r['question_id'],
                    statement=text, tests=tests, references=(r['4o_solution'].strip() + '\n',),
                    difficulty=r['gpt_difficulty'], time_limit_s=4.0, cluster_id=pid, validation_status='unverified',
                    metadata={'split_role': 'rl' if i < n_pool else 'probe', 'license': 'CC-BY-NC-4.0',
                              'gpt_pass_percentage': r['gpt_pass_percentage'], 'task': 'function'})
        problems.append(p.to_dict())
        refs.append({'request_id': digest(['kodcode-reference', pid])[:32], 'problem_id': pid, 'candidate_index': 0,
                     'text': '```python\n' + p.references[0] + '```', 'finish_reason': 'stop', 'output_tokens': 0,
                     'tests_hash': p.tests_hash, 'test_count': len(p.tests)})
    out.mkdir(parents=True)
    write_jsonl(out/'vm-problems-unverified.jsonl', problems)
    write_jsonl(out/'reference-cands.jsonl', refs)
    write_json(out/'build.json', {'created_at': utc_now(), 'shard_sha256': SHARD_SHA256, 'eligible': len(cand),
        'pool': n_pool, 'probe': n_probe, 'decontam_sha256': file_digest(DECONTAM),
        'files': {n: file_digest(out/n) for n in ('vm-problems-unverified.jsonl', 'reference-cands.jsonl')}})
    print(json.dumps({'eligible': len(cand), 'written': len(problems)}))


def build_dev(out, n_dev, role='dev', prior=()):
    """Held-out KodCode set (`dev`, or `test` for the final evaluation): the next eligible problems after
    the RL pool/probe and any `prior` held-out sets (same filters and seed-42 order), excluding Evol
    variants that share a metadata seed id with anything already used."""
    dev_dir = out/role
    if dev_dir.exists():
        raise ValueError('immutable_output_exists')
    if file_digest(SHARD) != SHARD_SHA256:
        raise ValueError('pinned_shard_changed')
    import pyarrow.parquet as pq
    bad = set()
    for line in DECONTAM.read_text(encoding='utf-8').splitlines():
        bad |= grams(json.loads(line)['text'])
    cand = list(eligible(pq.read_table(SHARD).to_pylist(), bad))
    cand.sort(key=lambda x: x[0]['question_id'])
    random.Random(42).shuffle(cand)
    used_rows = [json.loads(l) for l in (out/'vm-problems-unverified.jsonl').read_text(encoding='utf-8').splitlines()]
    used_ids = {r['source_id'] for r in used_rows}
    if [r['question_id'] for r, _ in cand[:len(used_ids)]] != [r['source_id'] for r in used_rows]:
        raise ValueError('eligible_order_changed_since_build')
    # Evol variants of one seed share metadata seed_ids; a plain 10-gram test mostly hits template
    # phrasing (63% of the remaining pool), so variants are excluded by seed id instead.
    used_seeds = set()
    for r, _ in cand[:len(used_ids)]:
        used_seeds |= seed_ids(r)
    by_id = {r['question_id']: r for r, _ in cand}
    for name in prior:
        for line in (out/name/'vm-problems-unverified.jsonl').read_text(encoding='utf-8').splitlines():
            sid = json.loads(line)['source_id']; used_ids.add(sid); used_seeds |= seed_ids(by_id[sid])
    problems, refs, overlap = [], [], 0
    for r, text in cand:
        if len(problems) == n_dev:
            break
        if r['question_id'] in used_ids:
            continue
        if not seed_ids(r) or seed_ids(r) & used_seeds:
            overlap += 1
            continue
        token = ''.join(secrets.choice(ALPHABET) for _ in range(24))
        pid = FUNCTION_TASK_PREFIX + r['question_id']
        tests = tuple(TestCase(stdin=s, stdout=o) for s, o in hidden_programs(r['test_code'], token))
        p = Problem(problem_id=pid, source='kodcode-v1-sft-4o', revision='14f8782f', source_id=r['question_id'],
                    statement=text, tests=tests, references=(r['4o_solution'].strip() + '\n',),
                    difficulty=r['gpt_difficulty'], time_limit_s=4.0, cluster_id=pid, validation_status='unverified',
                    metadata={'split_role': role, 'license': 'CC-BY-NC-4.0',
                              'gpt_pass_percentage': r['gpt_pass_percentage'], 'task': 'function'})
        problems.append(p.to_dict())
        refs.append({'request_id': digest(['kodcode-reference', pid])[:32], 'problem_id': pid, 'candidate_index': 0,
                     'text': '```python\n' + p.references[0] + '```', 'finish_reason': 'stop', 'output_tokens': 0,
                     'tests_hash': p.tests_hash, 'test_count': len(p.tests)})
    dev_dir.mkdir(parents=True)
    write_jsonl(dev_dir/'vm-problems-unverified.jsonl', problems)
    write_jsonl(dev_dir/'reference-cands.jsonl', refs)
    write_json(dev_dir/'build.json', {'created_at': utc_now(), 'shard_sha256': SHARD_SHA256, 'skipped_used': len(used_ids),
        'skipped_seed_overlap_with_used': overlap, 'written': len(problems),
        'files': {n: file_digest(dev_dir/n) for n in ('vm-problems-unverified.jsonl', 'reference-cands.jsonl')}})
    print(json.dumps({'written': len(problems), 'skipped_overlap': overlap}))


def finalize_dev(out, records):
    dev_dir = out/'dev'
    rows = [json.loads(l) for l in (dev_dir/'vm-problems-unverified.jsonl').read_text(encoding='utf-8').splitlines()]
    passed = {json.loads(f.read_text(encoding='utf-8'))['problem_id'] for f in Path(records).glob('*.json')
              if (lambda r: r.get('state') == 'complete' and r['binary_success'] and r['passed_tests'] == r['total_tests'])(
                  json.loads(f.read_text(encoding='utf-8')))}
    admitted, prompts = [], []
    for row in rows:
        if row['problem_id'] not in passed or 'def test_' in row['statement']:
            continue
        p = Problem.from_dict({**row, 'validation_status': 'reference_passed'})
        admitted.append(p.to_dict())
        prompts.append(checked_prompt({'problem_id': p.problem_id,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': p.statement}],
            'tests_hash': p.tests_hash, 'test_count': len(p.tests), 'difficulty': p.difficulty,
            'cluster_id': p.cluster_id, 'split': 'development'}))
    write_jsonl(dev_dir/'vm-dev-problems.jsonl', admitted)
    write_jsonl(dev_dir/'gpu-dev-prompts.jsonl', prompts)
    write_json(dev_dir/'finalize.json', {'created_at': utc_now(), 'verified': len(passed), 'admitted': len(admitted),
        'files': {n: file_digest(dev_dir/n) for n in ('vm-dev-problems.jsonl', 'gpu-dev-prompts.jsonl')}})
    print(json.dumps({'admitted': len(admitted)}))


def finalize(out, records):
    build_info = json.loads((out/'build.json').read_text(encoding='utf-8'))
    if file_digest(out/'vm-problems-unverified.jsonl') != build_info['files']['vm-problems-unverified.jsonl']:
        raise ValueError('unverified_rows_changed')
    rows = [json.loads(l) for l in (out/'vm-problems-unverified.jsonl').read_text(encoding='utf-8').splitlines()]
    passed = set()
    for f in Path(records).glob('*.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('state') == 'complete' and r['binary_success'] and r['passed_tests'] == r['total_tests']:
            passed.add(r['problem_id'])
    admitted, prompts = [], {'rl': [], 'probe': []}
    for row in rows:
        # Statements that quote test functions may restate hidden tests; keep tests strictly hidden.
        if row['problem_id'] not in passed or 'def test_' in row['statement']:
            continue
        p = Problem.from_dict({**row, 'validation_status': 'reference_passed'})
        admitted.append(p.to_dict())
        prompts[row['metadata']['split_role']].append(checked_prompt({'problem_id': p.problem_id,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': p.statement}],
            'tests_hash': p.tests_hash, 'test_count': len(p.tests), 'difficulty': p.difficulty,
            'cluster_id': p.cluster_id, 'split': 'train'}))
    write_jsonl(out/'vm-problems.jsonl', admitted)
    for role, items in prompts.items():
        write_jsonl(out/f'gpu-{role}-prompts.jsonl', items)
    write_json(out/'finalize.json', {'created_at': utc_now(), 'verified': len(passed), 'admitted': len(admitted),
        'rl': len(prompts['rl']), 'probe': len(prompts['probe']),
        'files': {n: file_digest(out/n) for n in ('vm-problems.jsonl', 'gpu-rl-prompts.jsonl', 'gpu-probe-prompts.jsonl')}})
    print(json.dumps({'admitted': len(admitted), 'rl': len(prompts['rl']), 'probe': len(prompts['probe'])}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('build', 'finalize', 'build-dev', 'finalize-dev', 'build-test'))
    p.add_argument('--output', type=Path, default=ROOT/'data/kodcode-rl-v1')
    p.add_argument('--pool', type=int, default=1200)
    p.add_argument('--probe', type=int, default=160)
    p.add_argument('--dev', type=int, default=360)
    p.add_argument('--test', type=int, default=310)
    p.add_argument('--records', type=Path, help='finalize stages: VM verification records directory')
    a = p.parse_args()
    {'build': lambda: build(a.output, a.pool, a.probe), 'finalize': lambda: finalize(a.output, a.records),
     'build-dev': lambda: build_dev(a.output, a.dev), 'finalize-dev': lambda: finalize_dev(a.output, a.records),
     'build-test': lambda: build_dev(a.output, a.test, 'test', ('dev',))}[a.stage]()


if __name__ == '__main__':
    main()
