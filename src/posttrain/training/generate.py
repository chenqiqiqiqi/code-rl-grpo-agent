"""Real vLLM generation for development/pilot/calibration; no final task access.

This is offline generation, NOT proof of verl in-place weight synchronization.
Interrupted batches retain their worst-case token debit until reconciled.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import math
import sys
import time
import uuid
import re
from pathlib import Path

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json,write_jsonl
from posttrain.data.prompt_manifest import checked_prompt
from posttrain.training.budget import TokenLedger,read_snapshot
from posttrain.training.sft import MODEL,checked_checkpoint
from posttrain.training.sft_data import chat_tokens


def checked_generation(cfg):
    penalty=cfg.get('repetition_penalty',1.0)
    if type(penalty) not in (int,float) or not math.isfinite(penalty) or penalty<=0:
        raise ValueError('invalid_repetition_penalty')
    if cfg.get('model_id')!=MODEL or not re.fullmatch('[a-f0-9]{40}',cfg.get('model_revision','')):raise ValueError('pinned_7b_model_required')
    if cfg.get('dtype')!='bfloat16':raise ValueError('BF16_required')
    for key in ('batch_prompts','g','max_prompt_tokens','max_new_tokens','token_cap','seed'):
        if type(cfg.get(key)) is not int or cfg[key]<(0 if key=='seed' else 1):raise ValueError('invalid_'+key)
    for key in ('temperature','top_p','gpu_memory_utilization'):
        if type(cfg.get(key)) not in (int,float) or not math.isfinite(cfg[key]):raise ValueError('invalid_'+key)
    if cfg['temperature']<0 or not 0<cfg['top_p']<=1 or not 0<cfg['gpu_memory_utilization']<1:
        raise ValueError('invalid_sampling_configuration')
    if cfg['temperature']==0 and cfg['g']!=1:raise ValueError('greedy_generation_requires_g_1')
    if type(cfg.get('top_k')) is not int or (cfg['top_k']!=-1 and cfg['top_k']<1):raise ValueError('invalid_top_k')
    return cfg


def sampling_arguments(cfg):
    return dict(n=cfg['g'],temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],
        repetition_penalty=cfg.get('repetition_penalty',1.0),
        max_tokens=cfg['max_new_tokens'],logprobs=1,seed=cfg['seed'])


def validate_generated(rows,prompts,g,max_tokens):
    if len(rows)!=len(prompts)*g:raise ValueError('incomplete_generation_batch')
    for i,prompt in enumerate(prompts):
        for j in range(g):
            row=rows[i*g+j]
            if row['problem_id']!=prompt['problem_id'] or row['candidate_index']!=j:
                raise ValueError('generation_prompt_or_candidate_order_changed')
            if len(row['token_ids'])>max_tokens or row['output_tokens']!=len(row['token_ids']):
                raise ValueError('output_token_count_mismatch')
            if len(row['sampled_token_log_probs'])!=row['output_tokens']:
                raise ValueError('missing_vllm_sampled_token_log_probs')
            if any(type(x) not in (float,int) or not math.isfinite(x) for x in row['sampled_token_log_probs']):
                raise ValueError('nonfinite_sampled_token_log_prob')


def checked_adapter_origin(adapter,cfg):
    manifest=checked_checkpoint(adapter)
    identity=json.loads((adapter.parent/'identity.json').read_text(encoding='utf-8'))
    if manifest.get('training_identity_sha256')!=digest(identity):raise ValueError('adapter_training_identity_mismatch')
    for key in ('model_id','model_revision'):
        if identity['config'][key]!=cfg[key]:raise ValueError('adapter_base_revision_mismatch')
    return manifest


def checked_rl_adapter(folder,cfg):
    """Adapter exported by scripts/export_rl_adapter.py from a receipt-verified verl checkpoint."""
    record=json.loads((folder/'export.json').read_text(encoding='utf-8'))
    if (record.get('scope')!='rl_lora_adapter_export'
        or record.get('base')!={'model_id':cfg['model_id'],'model_revision':cfg['model_revision']}):
        raise ValueError('rl_adapter_scope_or_base_mismatch')
    if set(record.get('files',{}))!={'adapter/adapter_config.json','adapter/adapter_model.safetensors'}:
        raise ValueError('rl_adapter_file_set_changed')
    if any(file_digest(folder/name)!=sha for name,sha in record['files'].items()):
        raise ValueError('rl_adapter_changed_since_export')
    if type(record.get('r')) is not int or not 1<=record['r']<=32:raise ValueError('rl_adapter_rank_exceeds_max_lora_rank')
    return {**record,'export_sha256':file_digest(folder/'export.json')}


def checked_generation_bundle(folder,prompts,prompts_hash):
    identity=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if (report.get('evidence_level')!='real_7b_offline_generation' or report.get('complete') is not True
        or report.get('identity_sha256')!=file_digest(folder/'identity.json')
        or report.get('candidates_sha256')!=file_digest(folder/'candidates.jsonl')
        or identity['prompts_sha256']!=prompts_hash):raise ValueError('generation_artifact_identity_mismatch')
    rows=list(read_jsonl(folder/'candidates.jsonl'));cfg=identity['configuration']
    validate_generated(rows,prompts,cfg['g'],cfg['max_new_tokens'])
    ledger=read_snapshot(folder/'token-ledger.sqlite')
    if ledger!=report['token_accounting'] or ledger['cap']!=cfg['token_cap']:
        raise ValueError('generation_ledger_changed')
    allocations={}
    for i,row in enumerate(rows):
        expected_id=digest([identity['run_uuid'],i//cfg['g'],i%cfg['g']])[:32]
        if row['request_id']!=expected_id:raise ValueError('candidate_request_identity_mismatch')
        allocations.setdefault(row['reservation'],[]).append(row)
    if set(allocations)!={r['id'] for r in ledger['reservations']}:raise ValueError('generation_reservation_set_mismatch')
    for reservation in ledger['reservations']:
        batch=allocations[reservation['id']]
        if (reservation['actual']!=sum(r['output_tokens'] for r in batch)
            or reservation['payload']['prompts']*reservation['payload']['g']!=len(batch)):
            raise ValueError('candidate_token_ledger_mismatch')
    return identity,report,rows


def run(config_path,prompt_path,output,adapter=None,model_snapshot=None,rl_adapter=None):
    cfg=checked_generation(json.loads(config_path.read_text(encoding='utf-8')))
    prompts=[checked_prompt(row) for row in read_jsonl(prompt_path)]
    if not prompts or len({p['problem_id'] for p in prompts})!=len(prompts):raise ValueError('empty_or_duplicate_prompts')
    if cfg['token_cap']<len(prompts)*cfg['g']*cfg['max_new_tokens']:
        raise ValueError('reserve_entire_preparation_job_token_maximum')
    if adapter and rl_adapter:raise ValueError('one_adapter_source_only')
    adapter_manifest=(checked_adapter_origin(adapter,cfg) if adapter else
                      checked_rl_adapter(rl_adapter,cfg) if rl_adapter else None)
    adapter=adapter or rl_adapter  # both layouts keep the PEFT files under <folder>/adapter
    model_source=MODEL
    model_revision=cfg['model_revision']
    snapshot_manifest_sha256=None
    if model_snapshot is not None:
        from posttrain.training.verl_pilot import ROOT,checked_model_files
        model_manifest=ROOT/'manifests/qwen7b_model_files.json'
        checked_model_files(model_snapshot,json.loads(model_manifest.read_text(encoding='utf-8')))
        model_source=str(model_snapshot.resolve())
        model_revision=None
        snapshot_manifest_sha256=file_digest(model_manifest)
    if sys.platform!='linux':raise RuntimeError('Linux_CUDA_required')
    import torch
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('real_CUDA_BF16_required')
    output.mkdir(parents=True,exist_ok=False)
    run_uuid=uuid.uuid4().hex
    identity={'run_uuid':run_uuid,'model_id':MODEL,'model_revision':cfg['model_revision'],'configuration':cfg,
        'prompts_sha256':file_digest(prompt_path),'adapter':adapter_manifest,
        'model_snapshot_manifest_sha256':snapshot_manifest_sha256,
        'model_load_source':model_source,
        'vllm_use_flashinfer_sampler':os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'),
        'source_sha256':file_digest(Path(__file__)),
        'tokenization_source_sha256':file_digest(Path(__file__).with_name('sft_data.py')),
        'created_at':utc_now(),'purpose':sorted({p['split'] for p in prompts}),
        'software':{k:importlib.metadata.version(k) for k in ('vllm','torch','transformers')}}
    write_json(output/'identity.json',identity)
    ledger=TokenLedger(output/'token-ledger.sqlite',cfg['token_cap'])
    started=time.monotonic()
    llm=LLM(model=model_source,revision=model_revision,tokenizer=model_source,
        tokenizer_revision=model_revision,
        dtype='bfloat16',trust_remote_code=False,tensor_parallel_size=1,seed=cfg['seed'],
        enable_lora=adapter is not None,max_lora_rank=32,enforce_eager=True,
        max_model_len=cfg['max_prompt_tokens']+cfg['max_new_tokens'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'])
    initialization_s=time.monotonic()-started
    tokenizer=llm.get_tokenizer()
    encoded=[chat_tokens(tokenizer,p['messages'],add_generation_prompt=True) for p in prompts]
    if any(len(x)>cfg['max_prompt_tokens'] for x in encoded):
        raise ValueError('prompt_exceeds_preregistered_context_no_silent_truncation')
    sampling=SamplingParams(**sampling_arguments(cfg))
    request=LoRARequest('project_sft',1,str((adapter/'adapter').resolve())) if adapter else None
    all_rows=[];batches=[]
    for offset in range(0,len(prompts),cfg['batch_prompts']):
        subset=prompts[offset:offset+cfg['batch_prompts']]
        reservation=ledger.reserve(len(subset),cfg['g'],cfg['max_new_tokens'],'offline_preparation_generation')
        batch_started=time.monotonic()
        generated=llm.generate([{'prompt_token_ids':x} for x in encoded[offset:offset+len(subset)]],
            sampling,lora_request=request,use_tqdm=False)
        wall=time.monotonic()-batch_started
        if len(generated)!=len(subset):raise ValueError('vllm_missing_prompt_output')
        rows=[]
        for i,(p,result) in enumerate(zip(subset,generated)):
            if list(result.prompt_token_ids)!=list(encoded[offset+i]):raise ValueError('vllm_prompt_token_mismatch')
            if len(result.outputs)!=cfg['g']:raise ValueError('vllm_missing_candidate')
            for j,candidate in enumerate(sorted(result.outputs,key=lambda x:x.index)):
                if candidate.index!=j:raise ValueError('vllm_candidate_index_mismatch')
                token_ids=list(candidate.token_ids)
                log_probs=[float(values[token].logprob) for token,values in zip(token_ids,candidate.logprobs or [])]
                rows.append({'request_id':digest([run_uuid,offset+i,j])[:32],'problem_id':p['problem_id'],
                    'candidate_index':j,'text':candidate.text,'token_ids':token_ids,'output_tokens':len(token_ids),
                    'input_tokens':len(encoded[offset+i]),'sampled_token_log_probs':log_probs,
                    'log_prob_source':'vLLM_RequestOutput_sampled_token_logprob_not_training_old_log_prob',
                    'finish_reason':candidate.finish_reason,'stop_reason':candidate.stop_reason,
                    'reservation':reservation,'tests_hash':p['tests_hash'],'test_count':p['test_count']})
        validate_generated(rows,subset,cfg['g'],cfg['max_new_tokens'])
        # Persist generation evidence before reducing the conservative debit.
        write_jsonl(output/'batches'/f'{offset:06d}.jsonl',rows)
        ledger.settle(reservation,sum(r['output_tokens'] for r in rows))
        batches.append({'offset':offset,'prompts':len(subset),'generation_wall_s':wall,
            'output_tokens':sum(r['output_tokens'] for r in rows),'reservation':reservation})
        all_rows.extend(rows);write_json(output/'timing.json',batches)
    write_jsonl(output/'candidates.jsonl',all_rows)
    report={'evidence_level':'real_7b_offline_generation','complete':True,'model_id':MODEL,
        'identity_sha256':file_digest(output/'identity.json'),'candidates_sha256':file_digest(output/'candidates.jsonl'),
        'prompts':len(prompts),'candidates':len(all_rows),'initialization_s':initialization_s,
        'total_wall_s':time.monotonic()-started,'token_accounting':ledger.snapshot(),
        'real_verl_weight_sync_validated':False,'training_updates':0,'final_benchmark_tasks':0}
    write_json(output/'report.json',report);ledger.close()
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sft-checkpoint',type=Path)
    p.add_argument('--rl-adapter',type=Path,help='scripts/export_rl_adapter.py output folder')
    p.add_argument('--model-snapshot',type=Path)
    a=p.parse_args();print(json.dumps(run(a.config,a.prompts,a.output,a.sft_checkpoint,a.model_snapshot,a.rl_adapter)))


if __name__=='__main__':main()
