"""Whitelisted GPU inputs. Frozen answers/references stay on the CPU scorer."""
from __future__ import annotations

import argparse
from pathlib import Path

from posttrain.common import read_jsonl,write_jsonl
from posttrain.data.schema import Problem

FIELDS={'problem_id','messages','tests_hash','test_count','difficulty','cluster_id','split'}


def public_prompt(problem,split):
    if problem.validation_status!='reference_passed':raise ValueError('reference_backtest_required')
    if split not in ('train','development','calibration','pilot'):raise ValueError('no_final_benchmark_generation_in_preparation')
    row={'problem_id':problem.problem_id,'messages':problem.prompt(),'tests_hash':problem.tests_hash,
        'test_count':len(problem.tests),'difficulty':problem.difficulty,'cluster_id':problem.cluster_id,'split':split}
    checked_prompt(row)
    return row


def checked_prompt(row):
    if set(row)!=FIELDS:raise ValueError('unexpected_GPU_input_fields')
    if row['split'] not in ('train','development','calibration','pilot'):raise ValueError('preparation_split_only')
    if not row['problem_id'] or not row['cluster_id'] or not row['tests_hash']:raise ValueError('missing_prompt_identity')
    if type(row['test_count']) is not int or row['test_count']<1:raise ValueError('empty_frozen_tests')
    messages=row['messages']
    if not isinstance(messages,list) or len(messages)!=2:raise ValueError('single_turn_messages_required')
    for msg,role in zip(messages,('system','user')):
        if set(msg)!= {'role','content'} or msg['role']!=role or not isinstance(msg['content'],str) or not msg['content']:
            raise ValueError('invalid_prompt_message')
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True,
        choices=['train','development','calibration','pilot'])
    args=p.parse_args()
    if args.output.exists():raise ValueError('do_not_overwrite_prompt_manifest')
    write_jsonl(args.output,(public_prompt(Problem.from_dict(row),args.split) for row in read_jsonl(args.input)))


if __name__=='__main__':main()
