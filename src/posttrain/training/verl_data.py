"""Pilot-only public prompt export for the pinned verl RLHFDataset contract.

The rule reward ground_truth contains identifiers, never hidden inputs/answers or
reference code. Export is static preparation and does not approve a GPU launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain.common import canonical,file_digest,read_jsonl,utc_now,write_json
from posttrain.data.prompt_manifest import checked_prompt


def pilot_rows(train,development):
    if not train or not development:
        raise ValueError('nonempty_train_and_development_required')
    for rows,splits in ((train,{'train','pilot'}),(development,{'development'})):
        for row in rows:
            checked_prompt(row)
            if row['split'] not in splits:
                raise ValueError('unexpected_pilot_dataset_split')
        if len({r['problem_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate_prompt_id')
    if ({r['problem_id'] for r in train}&{r['problem_id'] for r in development}
        or {r['cluster_id'] for r in train}&{r['cluster_id'] for r in development}):
        raise ValueError('train_development_cluster_overlap')

    def convert(rows):
        return [{'data_source':'posttrain_stdio','ability':'code','prompt':row['messages'],
            'reward_model':{'style':'rule','ground_truth':canonical({
                k:row[k] for k in ('problem_id','tests_hash','test_count')})},
            'extra_info':{'index':index,**{k:row[k] for k in
                ('problem_id','tests_hash','test_count','difficulty','cluster_id','split')}}}
            for index,row in enumerate(rows)]
    return convert(train),convert(development)


def export_pilot(train_path,development_path,output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise ValueError('immutable_dataset_output_exists')
    paths={'train':train_path,'development':development_path}
    before={name:file_digest(path) for name,path in paths.items()}
    train,development=pilot_rows(list(read_jsonl(train_path)),list(read_jsonl(development_path)))
    if before!={name:file_digest(path) for name,path in paths.items()}:
        raise ValueError('prompt_manifest_changed_during_read')
    output.mkdir(parents=True)
    for name,rows in (('train',train),('development',development)):
        path=output/(name+'.parquet')
        pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
        if pq.read_table(path).to_pylist()!=rows:
            raise ValueError('parquet_roundtrip_changed_public_prompt')
    report={'created_at':utc_now(),'scope':'engineering_pilot_only',
        'evidence_level':'static_parquet_roundtrip_no_model_execution',
        'verl_revision':'1876b06d0a3e4e71e06230be10af14492ca8a75b',
        'source_manifest_sha256':before,
        'parquet_sha256':{name:file_digest(output/(name+'.parquet')) for name in paths},
        'train_prompts':len(train),'development_prompts':len(development),
        'hidden_tests_exported':False,'reference_code_exported':False,
        'tokenizer_length_checked':False,'formal_training_eligible':False,
        'model_generations':0,'model_updates':0,'final_benchmark_tasks':0,
        'remaining_gate':'real tokenizer length checks, CPU attestation and instrumented verl loop'}
    write_json(output/'manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_pilot(args.train,args.development,args.output),ensure_ascii=False))


if __name__=='__main__':main()
