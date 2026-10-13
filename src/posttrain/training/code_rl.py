"""Shared single-turn/Agent GRPO entry point for protocol v3.

Preview is CPU-only. GPU pilot and formal launch have separate evidence gates.
Reuses the pinned verl composition and audit hooks without altering upstream.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import uuid
import urllib.request

from posttrain.common import digest,file_digest,read_jsonl,write_json
from posttrain.training import verl_pilot as pilot


def checked_settings(cfg):
    if cfg.get('mode') not in ('single','agent') or cfg.get('reward_kind') not in ('binary','mixed'):
        raise ValueError('unknown_rl_mode_or_reward')
    if cfg.get('scoring_mode','full') not in ('full','binary_fail_fast') or (
            cfg.get('scoring_mode')=='binary_fail_fast' and cfg['reward_kind']!='binary'):
        raise ValueError('binary_fail_fast_requires_binary_reward')
    if cfg.get('scope') not in ('engineering_pilot_only','code_rl_v3'):
        raise ValueError('explicit_run_scope_required')
    for key in ('batch_prompts','g','max_prompt_tokens','max_response_tokens','max_generated_tokens','max_turn_tokens','token_cap','total_steps'):
        if type(cfg.get(key)) is not int or cfg[key]<1:raise ValueError('invalid_'+key)
    if cfg['g']<2 or not cfg['max_turn_tokens']<=cfg['max_generated_tokens']<=cfg['max_response_tokens']:
        raise ValueError('invalid_generation_budget')
    if cfg['scope']=='engineering_pilot_only' and cfg['total_steps']<22:
        raise ValueError('twenty_post_warmup_steps_required')
    if cfg['mode']=='single' and cfg['max_generated_tokens']!=cfg['max_response_tokens']:
        raise ValueError('single_turn_response_is_generation_budget')
    if cfg['mode']=='agent' and (cfg.get('max_tool_calls')!=3 or cfg['max_turn_tokens']*4<cfg['max_generated_tokens']):
        raise ValueError('v3_agent_limits')
    if cfg.get('seed')!=42 or cfg.get('invalid_scoring_policy')!='abort_run_no_numeric_zero':
        raise ValueError('v3_seed_and_invalid_reward_policy')
    # Optional v4 optimisation/checkpoint overrides; absent keys keep configs/verl_pilot.json.
    for key in ('learning_rate','kl_loss_coef'):
        if key in cfg and (type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or cfg[key]<=0):
            raise ValueError('invalid_'+key)
    for key in ('save_freq','checkpoint_keep'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key]<1):raise ValueError('invalid_'+key)
    if 'save_lora_only' in cfg and type(cfg['save_lora_only']) is not bool:
        raise ValueError('invalid_save_lora_only')
    # Phase 2 (W/C/A) starts from W's exported PEFT adapter; the reference stays M0 (adapter disabled).
    if 'init_lora_adapter' in cfg and (type(cfg['init_lora_adapter']) is not str or not cfg['init_lora_adapter'].startswith('/')):
        raise ValueError('init_lora_adapter_must_be_absolute_path')
    # Phase-2 equal generation budget B2: stop before the first rollout once completed updates have
    # generated >= this many model tokens (total_steps is then only a safety ceiling).
    if 'stop_generated_tokens' in cfg and (type(cfg['stop_generated_tokens']) is not int or cfg['stop_generated_tokens']<1
            or cfg['stop_generated_tokens']>cfg['token_cap']):
        raise ValueError('invalid_stop_generated_tokens')
    # Phase-2 linear LR warmup for the re-initialised optimizer (verl constant schedule with warmup;
    # its first step has factor 0/N). Absent or 0 keeps the plain constant schedule.
    if 'lr_warmup_steps' in cfg and (type(cfg['lr_warmup_steps']) is not int or cfg['lr_warmup_steps']<0):
        raise ValueError('invalid_lr_warmup_steps')
    return cfg


def compose(cfg,source,model,dataset,output,context_path,context_hash,resume=None):
    from omegaconf import OmegaConf
    base=json.loads((pilot.ROOT/'configs/verl_pilot.json').read_text(encoding='utf-8'))
    for key in ('batch_prompts','g','max_prompt_tokens','max_response_tokens','seed','total_steps'):
        base[key]=cfg[key]
    # Optional per-GPU memory settings; absent keys keep the verl_pilot.json (A800 80GB) values.
    for key in ('gpu_memory_utilization','max_num_seqs','ppo_max_token_len_per_gpu','actor_param_offload','actor_optimizer_offload'):
        if key in cfg:base[key]=cfg[key]
    # Optional v4 optimisation/checkpoint overrides, validated again by pilot.checked_config.
    for key in ('learning_rate','kl_loss_coef','save_freq','checkpoint_keep'):
        if key in cfg:base[key]=cfg[key]
    base['total_steps']=max(22,base['total_steps'])
    base['token_cap']=base['total_steps']*base['batch_prompts']*base['g']*base['max_response_tokens']
    config=pilot.compose_config(base,source,model,dataset,output,context_path,context_hash,resume)
    OmegaConf.update(config,'trainer.project_name','posttrain-code-agent-rl-v3')
    OmegaConf.update(config,'trainer.total_training_steps',cfg['total_steps'])
    OmegaConf.update(config,'trainer.total_epochs',cfg['total_steps'])
    if cfg.get('save_lora_only',False):
        # Pinned verl saves only adapter tensors (base stays the hash-bound snapshot) and
        # auto-detects LoRA-only state on resume; optimizer/extra/data state are unchanged.
        OmegaConf.update(config,'actor_rollout_ref.actor.checkpoint.save_lora_only',True,force_add=True)
    if cfg.get('init_lora_adapter'):
        # Pinned verl loads it with PeftModel.from_pretrained(is_trainable=True); optimizer starts fresh.
        OmegaConf.update(config,'actor_rollout_ref.model.lora_adapter_path',cfg['init_lora_adapter'],force_add=True)
    if cfg.get('lr_warmup_steps'):
        OmegaConf.update(config,'actor_rollout_ref.actor.optim.lr_warmup_steps',cfg['lr_warmup_steps'])
    # v3 formal runs use a hash-bound merged common start; v4 (M1 = M0) uses the official base.
    if cfg['mode']=='agent':
        loop_path=output/'agent-loop.yaml'
        loop=[{'name':'code_debug_agent','_target_':'posttrain.agent.verl_loop.CodeAgentLoop',
               'context_path':str(context_path.resolve()),'context_sha256':context_hash,
               'max_tool_calls':cfg['max_tool_calls'],'max_generated_tokens':cfg['max_generated_tokens'],
               'max_turn_tokens':cfg['max_turn_tokens']}]
        OmegaConf.save(OmegaConf.create(loop),loop_path)
        OmegaConf.update(config,'actor_rollout_ref.rollout.agent.agent_loop_config_path',str(loop_path.resolve()))
        OmegaConf.update(config,'actor_rollout_ref.rollout.agent.default_agent_loop','code_debug_agent')
    OmegaConf.resolve(config)
    return config


def check_prepared(root,mode):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('hidden_tests_exported') is not False:raise ValueError('public_GPU_inputs_required')
    required={f'{mode}-{split}.jsonl' for split in ('train','development')}
    required|={f'{mode}-parquet/{name}' for name in ('manifest.json','train.parquet','development.parquet')}
    if not required<=set(manifest.get('files',{})):
        raise ValueError('prepared_manifest_missing_required_files')
    for name,sha in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_digest(path)!=sha:
            raise ValueError('prepared_input_changed:'+name)
    prompts=[{'path':str((root/f'{mode}-{split}.jsonl').resolve()),
             'sha256':file_digest(root/f'{mode}-{split}.jsonl')} for split in ('train','development')]
    pilot.checked_dataset(root/f'{mode}-parquet',prompts)
    return prompts


def live_identity(endpoint):
    from urllib.parse import urlsplit
    p=urlsplit(endpoint)
    if p.scheme!='http' or p.hostname not in ('127.0.0.1','localhost','::1') or p.username or p.path!='' or p.query or p.fragment:
        raise ValueError('loopback_gateway_required')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint+'/identity',timeout=15) as response:return json.load(response)


def validate_v3_freeze(freeze,root,cfg,model_manifest):
    from posttrain.training.evidence import require_real_pilot
    if freeze.get('protocol')!='code_agent_rl_v3' or freeze.get('seeds')!=[42]:
        raise ValueError('v3_freeze_required')
    require_real_pilot(freeze['pilot_evidence'])
    if freeze.get('reward_formulas')!={'binary':'all_pass','mixed':[.5,.5]}:
        raise ValueError('reward_matrix_changed')
    if freeze.get('common_start_manifest_sha256')!=file_digest(model_manifest):
        raise ValueError('common_start_changed')
    if freeze.get('settings_sha256')!=digest(cfg):raise ValueError('unfrozen_run_settings')
    calibration=freeze.get('calibration',{})
    if calibration.get('problems')!=187 or calibration.get('evidence_level')!='isolated_execution' or calibration.get('common_start_manifest_sha256')!=file_digest(model_manifest):
        raise ValueError('bound_187_problem_calibration_required')
    for name,minimum in (('valid_group_fraction',.95),('binary_signal_group_fraction',.1)):
        value=calibration.get(name)
        if type(value) not in (int,float) or not math.isfinite(value) or not minimum<=value<=1:
            raise ValueError('reward_calibration_required')
    if freeze['calibration'].get('problem_ids_sha256')!=freeze.get('calibration_ids_sha256') or not freeze.get('calibration_ids_sha256'):
        raise ValueError('calibration_identity_required')
    if cfg['mode']=='agent' and freeze.get('agent_pilot_passed') is not True:
        raise ValueError('real_multiturn_pilot_required')
    if freeze.get('checkpoint_rule')!='last_complete_update_at_normal_token_budget_end':
        raise ValueError('checkpoint_rule_required')
    cost=freeze.get('total_forecast_including_spent_cny')
    if type(cost) not in (int,float) or not math.isfinite(cost) or not 0<cost<=1600:
        raise ValueError('reserve_400_cny_of_total_2000')
    if not freeze.get('artifacts'):raise ValueError('frozen_artifacts_required')
    for item in freeze['artifacts']:
        if file_digest(root/item['path'])!=item['sha256']:raise ValueError('frozen_artifact_changed')


def validate_v4_freeze(freeze,root,cfg,prepared):
    """Protocol v4 revision: M1 = M0, so the official base is the common start and the
    reference; the frozen RL problem pool replaces the v3 187-problem calibration."""
    if freeze.get('protocol')!='code_agent_rl_v4' or freeze.get('seeds')!=[42]:
        raise ValueError('v4_freeze_required')
    if (freeze.get('common_start')!='M0_official_base'
        or freeze.get('model_manifest_sha256')!=file_digest(root/'manifests/qwen7b_model_files.json')):
        raise ValueError('v4_common_start_is_official_M0')
    if freeze.get('reward_formulas')!={'binary':'all_pass','mixed':[.5,.5]}:
        raise ValueError('reward_matrix_changed')
    if freeze.get('settings_sha256')!=digest(cfg):raise ValueError('unfrozen_run_settings')
    pilot=freeze.get('pilot_evidence',{})
    if (pilot.get('evidence_level')!='real_7b_verl_4090d_pilot' or pilot.get('resume_passed') is not True
        or type(pilot.get('stable_complete_steps',{}).get(cfg['mode'])) is not int
        or pilot['stable_complete_steps'][cfg['mode']]<22):
        raise ValueError('v4_real_pilot_required')
    if not pilot.get('report') or file_digest(root/pilot['report'])!=pilot.get('report_sha256'):
        raise ValueError('pilot_report_changed')
    pool=freeze.get('problem_pool',{})
    if pool.get('evidence_level')!='isolated_execution' or not pool.get('path'):
        raise ValueError('frozen_problem_pool_required')
    if file_digest(root/pool['path'])!=pool.get('sha256'):raise ValueError('problem_pool_changed')
    ids=[row['problem_id'] for row in json.loads((root/pool['path']).read_text(encoding='utf-8'))['pool']]
    if len(set(ids))!=len(ids) or len(ids)!=pool.get('problems') or digest(sorted(ids))!=pool.get('ids_sha256'):
        raise ValueError('problem_pool_identity_changed')
    train=[row['problem_id'] for row in read_jsonl(prepared/f"{cfg['mode']}-train.jsonl")]
    if len(set(train))!=len(train) or sorted(train)!=sorted(ids):
        raise ValueError('prepared_train_set_is_not_the_frozen_pool')
    if cfg['mode']=='agent' and freeze.get('agent_pilot_passed') is not True:
        raise ValueError('real_multiturn_pilot_required')
    if freeze.get('checkpoint_rule')!='last_complete_update_at_normal_token_budget_end':
        raise ValueError('checkpoint_rule_required')
    cost=freeze.get('total_forecast_including_spent_cny')
    if type(cost) not in (int,float) or not math.isfinite(cost) or not 0<cost<=1600:
        raise ValueError('reserve_400_cny_of_total_2000')
    if not freeze.get('artifacts'):raise ValueError('frozen_artifacts_required')
    for item in freeze['artifacts']:
        if file_digest(root/item['path'])!=item['sha256']:raise ValueError('frozen_artifact_changed')


def main():
    p=argparse.ArgumentParser()
    for name in ('config','model-snapshot','prepared','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verl-source',type=Path)
    for name in ('gpu-probe','model-files','attestation','gateway-identity','tool-identity','freeze','resume-receipt'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--score-endpoint',default='http://127.0.0.1:8080')
    p.add_argument('--tool-endpoint',default='http://127.0.0.1:8081')
    p.add_argument('--preview',action='store_true')
    p.add_argument('--backend',choices=('verl','single_model'),default='verl')
    p.add_argument('--stop-after',type=int,help='Single-model pilot: stop after this complete update and save a checkpoint')
    a=p.parse_args();cfg=checked_settings(json.loads(a.config.read_text(encoding='utf-8')))
    if a.stop_after is not None and a.backend!='single_model':
        raise ValueError('stop_after_requires_single_model_backend')
    if a.backend=='single_model':
        from posttrain.training.single_model_runner import checked_single_model_settings
        checked_single_model_settings(cfg)
    if a.backend=='verl':
        if 'pilot_format_bonus' in cfg:
            raise ValueError('pilot_format_bonus_single_model_only')
        if a.verl_source is None:raise ValueError('verl_source_required_for_verl_backend')
        pilot.verify_source(a.verl_source)
    prompts=check_prepared(a.prepared,cfg['mode'])
    resume=None
    if a.resume_receipt:resume=Path(json.loads(a.resume_receipt.read_text(encoding='utf-8'))['checkpoint'])
    a.output.mkdir(parents=True,exist_ok=bool(resume))
    context_path=a.output/'reward-context.json'
    if resume:
        context=json.loads(context_path.read_text(encoding='utf-8'))
        if (context['prompt_manifests']!=prompts or context['reward_kind']!=cfg['reward_kind']
            or context['endpoint']!=a.score_endpoint or context['tool_endpoint']!=a.tool_endpoint or context['scope']!=cfg['scope']
            or context.get('scoring_mode','full')!=cfg.get('scoring_mode','full')):
            raise ValueError('resume_reward_context_changed')
    else:
        context={'scope':cfg['scope'],'run_uuid':uuid.uuid4().hex,'prompt_manifests':prompts,
                 'reward_kind':cfg['reward_kind'],'scoring_mode':cfg.get('scoring_mode','full'),
                 'endpoint':a.score_endpoint,
                 'tool_endpoint':a.tool_endpoint,'executor_version':'NOT_ATTESTED',
                 'audit_directory':str((a.output/'scoring').resolve())}
    if not a.preview:
        if sys.platform!='linux':raise RuntimeError('Linux_CUDA_required')
        if any(getattr(a,k) is None for k in ('model_files','gpu_probe','attestation','gateway_identity')):
            raise ValueError('GPU_model_and_executor_evidence_required')
        model_manifest=json.loads(a.model_files.read_text(encoding='utf-8'))
        if cfg['scope']=='engineering_pilot_only':
            if model_manifest!=json.loads((pilot.ROOT/'manifests/qwen7b_model_files.json').read_text(encoding='utf-8')):
                raise ValueError('official_model_manifest_required')
            pilot.checked_model_files(a.model_snapshot,model_manifest)
        else:
            if a.freeze is None:raise ValueError('formal_v3_freeze_required')
            freeze=json.loads(a.freeze.read_text(encoding='utf-8'))
            v4=freeze.get('protocol')=='code_agent_rl_v4'
            if v4:validate_v4_freeze(freeze,pilot.ROOT,cfg,a.prepared)
            else:validate_v3_freeze(freeze,pilot.ROOT,cfg,a.model_files)
            if freeze.get('prepared_manifest_sha256')!=file_digest(a.prepared/'manifest.json'):
                raise ValueError('unfrozen_prepared_inputs')
            if v4:
                if model_manifest!=json.loads((pilot.ROOT/'manifests/qwen7b_model_files.json').read_text(encoding='utf-8')):
                    raise ValueError('official_model_manifest_required')
                pilot.checked_model_files(a.model_snapshot,model_manifest)
            else:
                if (model_manifest.get('kind')!='merged_bf16_common_start' or model_manifest.get('model_id')!=pilot.MODEL
                    or model_manifest.get('base_revision')!=pilot.MODEL_REVISION or not model_manifest.get('files')
                    or 'config.json' not in model_manifest['files'] or not any(n.endswith('.safetensors') for n in model_manifest['files'])):
                    raise ValueError('merged_start_required')
                for name,sha in model_manifest['files'].items():
                    if Path(name).name!=name or file_digest(a.model_snapshot/name)!=sha:raise ValueError('merged_model_changed')
        probe=json.loads(a.gpu_probe.read_text(encoding='utf-8'))
        if not probe.get('passed') or not probe.get('bf16_kernel_and_backward'):raise ValueError('GPU_probe_required')
        from posttrain.execution.sandboxfusion import validate_attestation
        from posttrain.execution.contracts import Limits
        record=json.loads(a.attestation.read_text(encoding='utf-8'))
        validate_attestation(a.attestation,record['endpoint'],Limits(time_s=30))
        gateway=live_identity(a.score_endpoint)
        if gateway!=json.loads(a.gateway_identity.read_text(encoding='utf-8')):raise ValueError('scorer_changed')
        version='sandboxfusion-docker:'+record['image_digest']+':'+record['deployment_id']
        if gateway['executor_version']!=version:raise ValueError('scorer_attestation_mismatch')
        if resume and context['executor_version']!=version:raise ValueError('resume_executor_changed')
        context['executor_version']=version
        if cfg['mode']=='agent':
            tool=live_identity(a.tool_endpoint)
            if a.tool_identity is None or tool!=json.loads(a.tool_identity.read_text(encoding='utf-8')) or tool['executor_version']!=version:
                raise ValueError('tool_attestation_mismatch')
        import torch
        if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
            raise ValueError('single_bf16_GPU_runtime_required')
        if a.backend=='verl':
            import verl
            if Path(verl.__file__).resolve().parent!=(a.verl_source/'verl').resolve():
                raise ValueError('pinned_verl_import_required')
    write_json(context_path,context)
    if a.backend=='verl':
        config=compose(cfg,a.verl_source,a.model_snapshot,a.prepared/f"{cfg['mode']}-parquet",a.output,
                       context_path,file_digest(context_path),resume)
        from omegaconf import OmegaConf
        resolved=OmegaConf.to_container(config,resolve=True)
    else:
        resolved={'backend':'single_model','scope':'engineering_pilot_only','settings':cfg,
                  'model_copies':1,'update_epochs':2,'clip_ratio':0.2,'kl_coef':0.001,
                  'context_sha256':file_digest(context_path)}
    write_json(a.output/('preview.json' if a.preview else 'resolved-config.json'),resolved)
    if a.preview:
        print('Configuration composed; no model execution.');return
    settings={**cfg,'loss_agg_mode':'token-mean','warmup_observation_steps':2}
    bound={'settings':cfg,'prepared':file_digest(a.prepared/'manifest.json'),
           'model':file_digest(a.model_files),'context':file_digest(context_path),
           'source':{str(p.relative_to(pilot.ROOT)):file_digest(p) for p in sorted((pilot.ROOT/'src/posttrain').rglob('*.py'))}}
    if a.freeze is not None:bound['freeze']=file_digest(a.freeze)  # resume must keep the same freeze
    identity={'scope':cfg['scope'],'run_uuid':context['run_uuid'],'backend':a.backend,
              'config_sha256':digest(bound)}
    import fcntl
    with (a.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.backend=='verl':
            pilot.run_trainer(config,settings,identity,a.output,a.resume_receipt)
        else:
            from posttrain.training.single_model_runner import run
            run(cfg,a.model_snapshot,context_path,file_digest(context_path),a.output,identity,
                a.resume_receipt,stop_after=a.stop_after)


if __name__=='__main__':main()
