"""Restrict prepared GPU prompts to a frozen RL problem pool (protocol v4, Stage 4).

Keeps each selected train line byte-identical and in the source order; development
lines and the pool file are copied unchanged. CPU only: no model, no execution.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from posttrain.common import digest,file_digest,utc_now,write_json
from posttrain.training.verl_data import export_pilot


def pool_ids(pool_path):
    ids=[row['problem_id'] for row in json.loads(pool_path.read_text(encoding='utf-8'))['pool']]
    if not ids or len(set(ids))!=len(ids):raise ValueError('empty_or_duplicate_pool')
    return ids


def select_lines(source,ids):
    wanted=set(ids);kept=[];seen=set()
    with source.open('rb') as stream:
        for line in stream:
            if not line.strip():continue
            pid=json.loads(line)['problem_id']
            if pid in wanted:
                if pid in seen:raise ValueError('duplicate_source_problem:'+pid)
                seen.add(pid);kept.append(line if line.endswith(b'\n') else line+b'\n')
    if seen!=wanted:raise ValueError(f'pool_problems_missing_from_source:{len(wanted-seen)}')
    return kept


def prepare(source,pool,output,modes):
    if output.exists():raise ValueError('immutable_preparation_output_exists')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('hidden_tests_exported') is not False:raise ValueError('public_GPU_inputs_required')
    for mode in modes:
        for split in ('train','development'):
            name=f'{mode}-{split}.jsonl'
            if manifest['files'].get(name)!=file_digest(source/name):raise ValueError('source_changed:'+name)
    ids=pool_ids(pool)
    output.mkdir(parents=True)
    shutil.copyfile(pool,output/'rl-problem-pool.json')
    for mode in modes:
        (output/f'{mode}-train.jsonl').write_bytes(b''.join(select_lines(source/f'{mode}-train.jsonl',ids)))
        shutil.copyfile(source/f'{mode}-development.jsonl',output/f'{mode}-development.jsonl')
        export_pilot(output/f'{mode}-train.jsonl',output/f'{mode}-development.jsonl',output/f'{mode}-parquet')
    if file_digest(output/'rl-problem-pool.json')!=file_digest(pool):raise ValueError('pool_copy_changed')
    write_json(output/'manifest.json',{'scope':'rl_problem_pool_preparation_not_training_acceptance',
        'created_at':utc_now(),'modes':list(modes),
        'source':{'path':source.resolve().relative_to(ROOT).as_posix(),'manifest_sha256':file_digest(source/'manifest.json')},
        'pool':{'path':pool.resolve().relative_to(ROOT).as_posix(),'sha256':file_digest(pool),
                'problems':len(ids),'ids_sha256':digest(sorted(ids))},
        'files':{p.relative_to(output).as_posix():file_digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'hidden_tests_exported':False,'reference_code_exported':False,
        'model_updates':0,'model_generations':0,'final_benchmark_tasks':0})
    return {'problems':len(ids),'output':str(output)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'data/prepared/agent-rl-v5b')
    p.add_argument('--pool',type=Path,default=ROOT/'eval-baseline-20260930/rl-problem-pool.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/prepared/rl-pool-v5b')
    p.add_argument('--modes',nargs='+',choices=('single','agent'),default=['single'])
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.pool,a.output,a.modes),ensure_ascii=False))


if __name__=='__main__':main()
