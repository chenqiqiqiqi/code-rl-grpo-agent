"""Remote execution reward bridge for engineering pilot only.

Use through pinned verl's custom_reward_function interface. Infrastructure
failures abort the pilot; no invalid score becomes a numeric reward. This bridge
does not implement formal whole-group replacement or token-budget control.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import time
import urllib.error
import uuid

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json
from posttrain.data.prompt_manifest import checked_prompt
from posttrain.evaluation.extraction import extract_python
from posttrain.execution.function_harness import is_function_task,wrap as wrap_function_candidate
from posttrain.execution.service import ScoringClient
from posttrain.rewards import rewards


class PilotScoringFailure(RuntimeError):
    """Distinct from TimeoutError: some legacy managers convert it to zero."""


def check_input(data_source,ground_truth,extra_info,prompt):
    checked_prompt(prompt)
    if data_source!='posttrain_stdio' or prompt['split'] not in ('train','development','pilot'):
        raise ValueError('pilot_stdio_source_required')
    identity={k:prompt[k] for k in ('problem_id','tests_hash','test_count')}
    if json.loads(ground_truth)!=identity:
        raise ValueError('reward_identity_mismatch')
    if any(extra_info.get(k)!=prompt[k] for k in ('problem_id','tests_hash','test_count','cluster_id','split')):
        raise ValueError('reward_metadata_changed')


def score_candidate(solution,prompt,client,output,request_id,reward_kind,context_id=None):
    if reward_kind not in ('binary','mixed') or not re.fullmatch(r'[a-f0-9]{32}',request_id):
        raise ValueError('invalid_pilot_reward_request')
    record_path=output/'records'/(request_id+'.json')
    claim=output/'claims'/(request_id+'.claim')
    claim.parent.mkdir(parents=True,exist_ok=True)
    if record_path.exists():raise ValueError('existing_request_requires_reconciliation')
    try:
        with claim.open('x',encoding='utf-8') as stream:stream.write(request_id)
    except FileExistsError:
        raise ValueError('existing_request_requires_reconciliation') from None
    code=extract_python(solution);format_rejected=code is None
    if format_rejected:code='def <POSTTRAIN_FORMAT_REJECTED>(\n'
    elif is_function_task(prompt['problem_id']):code=wrap_function_candidate(code)  # hidden tests arrive on stdin
    row={'created_at':utc_now(),'state':'pending','request_id':request_id,
        'problem_id':prompt['problem_id'],'tests_hash':prompt['tests_hash'],'test_count':prompt['test_count'],
        'raw_response_hash':digest(solution),'submitted_code':code,'code_hash':digest(code),
        'format_rejected':format_rejected,'context_sha256':context_id,'rewards':None,
        'scope':'pilot_scoring_callback_not_formal_training_acceptance'}
    write_json(record_path,row)
    started=time.monotonic();reconciled=False
    try:
        try:result=client.score(prompt['problem_id'],code,request_id)
        except (OSError,TimeoutError):
            # A failed transport may have executed already. GET only, never a
            # second POST. Unknown results abort and retain the durable intent.
            result=client.reconcile(prompt['problem_id'],code,request_id);reconciled=True
        row['result']=result.to_dict()
        if (not (result.binary_valid if reward_kind=='binary' else result.valid)
            or result.total!=prompt['test_count'] or result.tests_hash!=prompt['tests_hash']
            or result.code_hash!=digest(code) or result.request_id!=request_id
            or result.evidence_level!='isolated_execution' or (format_rejected and result.passed)):
            raise PilotScoringFailure('invalid_or_changed_frozen_judgment')
        binary_only=result.scoring_mode=='binary_fail_fast'
        values=rewards(result,binary_only=binary_only)
        if values is None or reward_kind not in values:raise PilotScoringFailure('reward_not_available')
        row.update(state='complete',rewards=values,reconciled=reconciled,completed_at=utc_now(),
            request_wall_s=time.monotonic()-started)
        write_json(record_path,row)
        return {'score':values[reward_kind],**values,'request_id':request_id,
            'mixed_available':not binary_only,
            'completed_tests':len(result.cases),'scoring_mode':result.scoring_mode,
            'passed_tests':result.passed,'total_tests':result.total,'format_rejected':format_rejected,
            'queue_s':result.queue_s,
            'judging_s':None if result.queue_s is None else max(0.,result.elapsed_s-result.queue_s),
            'request_wall_s':row['request_wall_s'],'physical_executions':result.physical_executions}
    except Exception as exc:
        row.update(state='unknown_or_invalid',rewards=None,failure_type=type(exc).__name__,
            request_wall_s=time.monotonic()-started)
        write_json(record_path,row)
        raise PilotScoringFailure('pilot_scoring_aborted_no_numeric_reward:'+request_id) from exc


@lru_cache(maxsize=2)
def load_context(config_path,config_sha256):
    path=Path(config_path)
    if file_digest(path)!=config_sha256:raise ValueError('pilot_reward_configuration_changed')
    cfg=json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('scope') not in ('engineering_pilot_only','code_rl_v3') or not re.fullmatch(r'[a-f0-9]{32}',cfg['run_uuid']):
        raise ValueError('pilot_reward_scope_required')
    prompts={}
    for artifact in cfg['prompt_manifests']:
        source=Path(artifact['path'])
        if file_digest(source)!=artifact['sha256']:raise ValueError('pilot_prompt_manifest_changed')
        for prompt in read_jsonl(source):
            checked_prompt(prompt)
            if prompt['split'] not in ('train','development','pilot'):raise ValueError('pilot_prompt_split_required')
            if prompt['problem_id'] in prompts:raise ValueError('duplicate_pilot_prompt')
            prompts[prompt['problem_id']]=prompt
    if not prompts or cfg['reward_kind'] not in ('binary','mixed'):raise ValueError('invalid_reward_context')
    client=ScoringClient(cfg['endpoint'],cfg['executor_version'],prompts,
        scoring_mode=cfg.get('scoring_mode','full'))
    if client.scoring_mode=='binary_fail_fast' and cfg['reward_kind']!='binary':
        raise ValueError('fail_fast_requires_binary_reward')
    return cfg,prompts,client


def compute_score(data_source,solution_str,ground_truth,extra_info,*,config_path,config_sha256):
    """Synchronous callbacks run in the pinned manager's executor thread."""
    cfg,prompts,client=load_context(config_path,config_sha256)
    prompt=prompts[extra_info['problem_id']]
    check_input(data_source,ground_truth,extra_info,prompt)
    # Every sampled candidate gets a distinct request, even when code is equal.
    return score_candidate(solution_str,prompt,client,Path(cfg['audit_directory']),uuid.uuid4().hex,
        cfg['reward_kind'],config_sha256)
