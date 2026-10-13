"""Offline development/calibration scoring through the attested CPU gateway."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import time
import urllib.error

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json,write_jsonl
from posttrain.data.prompt_manifest import checked_prompt
from posttrain.evaluation.extraction import extract_python
from posttrain.execution.contracts import CaseResult,JudgeResult,Status
from posttrain.execution.service import ScoringClient,ScoringOutcomeUnknown
from posttrain.rewards import group_statistics,rewards
from posttrain.training.generate import checked_generation_bundle


def from_record(row):
    data={f.name:row[f.name] for f in fields(JudgeResult)}
    data['status']=Status(data['status'])
    data['cases']=tuple(CaseResult(**{**c,'status':Status(c['status'])}) for c in data['cases'])
    result=JudgeResult(**data)
    if row['valid']!=result.valid:raise ValueError('inconsistent_persisted_score')
    return result


def summarize_scores(prompts,rows,g,temperature=None):
    if not prompts or len({p['problem_id'] for p in prompts})!=len(prompts):raise ValueError('empty_or_duplicate_prompts')
    expected={(p['problem_id'],i) for p in prompts for i in range(g)}
    keys=[(r['problem_id'],r['candidate_index']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('incomplete_or_duplicate_candidate_results')
    by_key={k:from_record(r['result']) for k,r in zip(keys,rows)}
    groups=[[by_key[(p['problem_id'],i)] for i in range(g)] for p in prompts]
    if g==1:
        if temperature is None:raise ValueError('single_draw_requires_known_decoding_temperature')
        good=[x[0] for x in groups if x[0].valid]
        report={'planned_tasks':len(prompts),'valid_tasks':len(good),'complete':len(good)==len(prompts),
            'decoding':'greedy' if temperature==0 else 'sampled_single_draw',
            'greedy_pass1' if temperature==0 else 'sampled_pass1':
                sum(x.passed==x.total for x in good)/len(prompts) if len(good)==len(prompts) else None}
    else:
        report=group_statistics(groups,g)
        report['by_difficulty']={level:group_statistics([group for p,group in zip(prompts,groups)
            if p['difficulty']==level],g) for level in sorted({p['difficulty'] for p in prompts})}
    duplicate=0
    for p in prompts:
        hashes=[r['code_hash'] for r in rows if r['problem_id']==p['problem_id']]
        duplicate+=len(hashes)-len(set(hashes))
    report.update({'problems':len(prompts),'candidates':len(rows),'duplicate_candidate_fraction':duplicate/len(rows),
        'format_rejected_candidates':sum(r['format_rejected'] for r in rows),
        'output_tokens':sum(r['output_tokens'] for r in rows)})
    return report


def score(generation_dir,prompt_path,output,endpoint,executor_version,workers):
    prompts=[checked_prompt(p) for p in read_jsonl(prompt_path)]
    identity,generation,candidates=checked_generation_bundle(generation_dir,prompts,file_digest(prompt_path))
    g=identity['configuration']['g']
    expected={(p['problem_id'],i) for p in prompts for i in range(g)}
    if len(candidates)!=len(expected) or {(r['problem_id'],r['candidate_index']) for r in candidates}!=expected:
        raise ValueError('incomplete_or_duplicate_generation')
    if type(workers) is not int or workers<1:raise ValueError('invalid_scoring_workers')
    output.mkdir(parents=True,exist_ok=False)
    by_id={p['problem_id']:p for p in prompts}
    client=ScoringClient(endpoint,executor_version,by_id)
    write_json(output/'identity.json',{'generation_identity_sha256':file_digest(generation_dir/'identity.json'),
        'candidates_sha256':file_digest(generation_dir/'candidates.jsonl'),'prompts_sha256':file_digest(prompt_path),
        'executor_version':executor_version,'workers':workers,'started_at':utc_now(),
        'source_sha256':file_digest(Path(__file__))})
    started=time.monotonic()
    def one(candidate):
        begin=time.monotonic();p=by_id[candidate['problem_id']]
        if candidate['tests_hash']!=p['tests_hash'] or candidate['test_count']!=p['test_count']:
            raise ValueError('candidate_frozen_tests_identity_changed')
        extracted=extract_python(candidate['text']);failed=extracted is None
        # Frozen format contract: ambiguous/missing code is a program failure.
        # Score a fixed syntax error in the real executor; explicitly record that
        # this sentinel, not a guessed model program, was submitted. No resampling.
        code=extracted if extracted is not None else 'def <POSTTRAIN_FORMAT_REJECTED>(\n'
        reconciled=False;failure=None
        try:
            try:result=client.score(p['problem_id'],code,candidate['request_id'])
            except urllib.error.HTTPError as exc:
                if exc.code!=409:raise
                result=client.reconcile(p['problem_id'],code,candidate['request_id']);reconciled=True
        except (OSError,TimeoutError,ValueError,ScoringOutcomeUnknown) as exc:
            failure=type(exc).__name__
            # Transport/identity failures are unknown, never a numeric reward.
            result=JudgeResult(candidate['request_id'],Status.INFRA,0,p['test_count'],(),
                time.monotonic()-begin,None,None,0,False,executor_version,'isolated_execution',digest(code),p['tests_hash'])
        row={'problem_id':p['problem_id'],'candidate_index':candidate['candidate_index'],
            'code_hash':digest(code),'raw_response_hash':digest(candidate['text']),'format_rejected':failed,
            'submitted_code_kind':'fixed_format_error_sentinel' if failed else 'extracted_model_code',
            'output_tokens':candidate['output_tokens'],'result':result.to_dict(),'rewards':rewards(result),
            'scoring_request_wall_s':time.monotonic()-begin,'transport_failure':failure,'reconciled_existing_result':reconciled}
        write_json(output/'records'/(candidate['request_id']+'.json'),row)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(one,candidates))
    write_jsonl(output/'scores.jsonl',rows)
    report=summarize_scores(prompts,rows,g,temperature=identity['configuration']['temperature'])
    common_start=digest({'model':identity['model_revision'],'adapter':identity.get('adapter')})
    report.update({'evidence_level':'isolated_execution','generation_evidence':'real_7b_offline_generation',
        'common_start':common_start,'generation_identity_sha256':file_digest(generation_dir/'identity.json'),
        'score_wall_s':time.monotonic()-started,'created_at':utc_now(),'executor_version':executor_version,
        'final_benchmark_tasks':0,'training_updates':0})
    report['formal_calibration_eligible']=bool(len(prompts)==200 and {p['split'] for p in prompts}=={'calibration'}
        and report.get('diagnostic_thresholds_met'))
    if g>1:
        report['binary_signal_group_fraction']=report['binary']['signal_group_fraction']
        report['mixed_signal_group_fraction']=report['mixed']['signal_group_fraction']
    write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--generation-dir',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',required=True);p.add_argument('--executor-version',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(json.dumps(score(a.generation_dir,a.prompts,a.output,a.endpoint,a.executor_version,a.workers)))


if __name__=='__main__':main()
