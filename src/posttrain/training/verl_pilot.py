"""Pinned single-GPU engineering pilot entry point, not a formal GRPO runner.

Configuration can be composed on Windows without importing Torch or Ray. Actual
launch requires Linux CUDA, authenticated model files and fresh scoring evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import urllib.request

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json

MODEL='Qwen/Qwen2.5-Coder-7B-Instruct'
MODEL_REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
VERL_REVISION='1876b06d0a3e4e71e06230be10af14492ca8a75b'
ROOT=Path(__file__).resolve().parents[3]


def checked_config(cfg):
    expected={'scope':'engineering_pilot_only','model_id':MODEL,'model_revision':MODEL_REVISION,
        'verl_revision':VERL_REVISION,'dtype':'bfloat16','formal_training_supported':False,
        'reference_policy':'M0_base_with_adapters_disabled',
        'invalid_scoring_policy':'abort_pilot_no_numeric_zero','loss_agg_mode':'token-mean'}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('fixed_engineering_pilot_only')
    positive=('total_steps','warmup_observation_steps','batch_prompts','g','max_prompt_tokens',
        'max_response_tokens','token_cap','lora_rank','lora_alpha','save_freq','checkpoint_keep',
        'max_num_seqs','reward_workers')
    if any(type(cfg.get(k)) is not int or cfg[k]<1 for k in positive):raise ValueError('positive_pilot_dimensions_required')
    if cfg['g']<2 or cfg['total_steps']-cfg['warmup_observation_steps']<20:
        raise ValueError('twenty_post_warmup_steps_and_group_sampling_required')
    if cfg['token_cap']<cfg['total_steps']*cfg['batch_prompts']*cfg['g']*cfg['max_response_tokens']:
        raise ValueError('insufficient_pilot_worst_case_token_cap')
    if type(cfg.get('seed')) is not int or cfg['seed']<0:raise ValueError('invalid_seed')
    for k in ('temperature','top_p','learning_rate','kl_loss_coef','gpu_memory_utilization'):
        if type(cfg.get(k)) not in (int,float) or not math.isfinite(cfg[k]) or cfg[k]<=0:
            raise ValueError('invalid_'+k)
    if cfg['top_p']!=1 or cfg.get('top_k')!=-1 or not cfg['gpu_memory_utilization']<1:
        raise ValueError('pilot_sampling_requires_unfiltered_distribution')
    # Optional memory settings (defaults reproduce the A800 80GB pilot); 48GB cards need them.
    for k in ('actor_param_offload','actor_optimizer_offload'):
        if k in cfg and type(cfg[k]) is not bool:raise ValueError('invalid_'+k)
    if 'ppo_max_token_len_per_gpu' in cfg and (type(cfg['ppo_max_token_len_per_gpu']) is not int
            or cfg['ppo_max_token_len_per_gpu']<cfg['max_prompt_tokens']+cfg['max_response_tokens']):
        raise ValueError('ppo_token_len_must_fit_one_full_sequence')
    return cfg


def checked_dataset(folder,prompt_manifests):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('scope')!='engineering_pilot_only' or manifest.get('verl_revision')!=VERL_REVISION:
        raise ValueError('pilot_dataset_revision_or_scope_changed')
    if len(prompt_manifests)!=2:raise ValueError('ordered_train_and_development_sources_required')
    for name,source in zip(('train','development'),prompt_manifests,strict=True):
        if file_digest(Path(source['path']))!=source['sha256'] or source['sha256']!=manifest['source_manifest_sha256'][name]:
            raise ValueError('pilot_prompt_source_changed')
        if file_digest(folder/(name+'.parquet'))!=manifest['parquet_sha256'][name]:
            raise ValueError('pilot_parquet_changed')
    return manifest


def checked_model_files(folder,manifest):
    if manifest.get('model_id')!=MODEL or manifest.get('revision')!=MODEL_REVISION or not manifest.get('files'):
        raise ValueError('fixed_model_file_manifest_required')
    for name,artifact in manifest['files'].items():
        path=folder/name
        if (Path(name).name!=name or not path.is_file() or path.stat().st_size!=artifact['bytes']
            or file_digest(path)!=artifact['sha256']):
            raise ValueError('model_file_missing_or_changed:'+name)
    if 'model.safetensors.index.json' in manifest['files']:
        index=json.loads((folder/'model.safetensors.index.json').read_text(encoding='utf-8'))
        shards=set(index['weight_map'].values())
        expected={name for name in manifest['files'] if name.endswith('.safetensors')}
        if not shards or shards!=expected:
            raise ValueError('model_index_references_unverified_weights')
        if {p.name for p in folder.glob('*.safetensors')}!=expected:
            raise ValueError('model_index_extra_weight_files')


def verify_source(source):
    record=json.loads((ROOT/'manifests/verl_source.json').read_text(encoding='utf-8'))
    archive=source.parent/'verl-source.tar.gz'
    if record['revision']!=VERL_REVISION or file_digest(archive)!=record['archive_sha256']:
        raise ValueError('pinned_verl_archive_required')
    spec=importlib.util.spec_from_file_location('posttrain_bootstrap_verify',ROOT/'scripts/bootstrap_gpu.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify_source_tree(archive,source)


def compose_config(cfg,source,model,dataset,output,reward_path,reward_sha256,resume=None):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    checked_config(cfg)
    with initialize_config_dir(version_base=None,config_dir=str((source/'verl/trainer/config').resolve())):
        config=compose(config_name='ppo_trainer')
    values={
        'data.train_files':[str((dataset/'train.parquet').resolve())],
        'data.val_files':[str((dataset/'development.parquet').resolve())],
        'data.train_batch_size':cfg['batch_prompts'],'data.max_prompt_length':cfg['max_prompt_tokens'],
        'data.max_response_length':cfg['max_response_tokens'],'data.seed':cfg['seed'],
        'data.dataloader_num_workers':0,'data.filter_overlong_prompts':False,'data.truncation':'error',
        'actor_rollout_ref.model.path':str(model.resolve()),
        'actor_rollout_ref.model.trust_remote_code':False,
        'actor_rollout_ref.model.lora_rank':cfg['lora_rank'],
        'actor_rollout_ref.model.lora_alpha':cfg['lora_alpha'],
        'actor_rollout_ref.model.target_modules':'all-linear',
        'actor_rollout_ref.model.enable_gradient_checkpointing':True,
        'actor_rollout_ref.model.use_remove_padding':True,
        'actor_rollout_ref.actor.strategy':'fsdp',
        'actor_rollout_ref.actor.ppo_mini_batch_size':cfg['batch_prompts'],
        'actor_rollout_ref.actor.use_dynamic_bsz':True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':cfg.get('ppo_max_token_len_per_gpu',2*(cfg['max_prompt_tokens']+cfg['max_response_tokens'])),
        'actor_rollout_ref.actor.loss_agg_mode':cfg['loss_agg_mode'],
        'actor_rollout_ref.actor.optim.lr':cfg['learning_rate'],
        'actor_rollout_ref.actor.optim.lr_scheduler_type':'constant',
        'actor_rollout_ref.actor.use_kl_loss':True,
        'actor_rollout_ref.actor.kl_loss_coef':cfg['kl_loss_coef'],
        'actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
        'actor_rollout_ref.actor.entropy_coeff':0.,
        'actor_rollout_ref.actor.data_loader_seed':cfg['seed'],
        'actor_rollout_ref.actor.fsdp_config.seed':cfg['seed'],
        'actor_rollout_ref.actor.fsdp_config.model_dtype':'bfloat16',
        'actor_rollout_ref.actor.fsdp_config.param_offload':cfg.get('actor_param_offload',False),
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':cfg.get('actor_optimizer_offload',False),
        'actor_rollout_ref.actor.checkpoint.save_contents':['model','optimizer','extra'],
        'actor_rollout_ref.actor.checkpoint.async_save':False,
        'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.n':cfg['g'],'actor_rollout_ref.rollout.seed':cfg['seed'],
        'actor_rollout_ref.rollout.temperature':cfg['temperature'],
        'actor_rollout_ref.rollout.top_p':cfg['top_p'],'actor_rollout_ref.rollout.top_k':cfg['top_k'],
        'actor_rollout_ref.rollout.dtype':'bfloat16','actor_rollout_ref.rollout.enforce_eager':True,
        'actor_rollout_ref.rollout.gpu_memory_utilization':cfg['gpu_memory_utilization'],
        'actor_rollout_ref.rollout.max_model_len':cfg['max_prompt_tokens']+cfg['max_response_tokens'],
        'actor_rollout_ref.rollout.max_num_batched_tokens':4096,
        'actor_rollout_ref.rollout.max_num_seqs':cfg['max_num_seqs'],
        'actor_rollout_ref.rollout.load_format':'safetensors',
        # Single-rank FSDP uses NO_SHARD. The pinned layered collector's empty
        # fallback requests unsupported CPU offload; full summon gathers the
        # adapters without that offload (the base is preloaded via safetensors).
        'actor_rollout_ref.rollout.layered_summon':False,
        'actor_rollout_ref.rollout.calculate_log_probs':True,
        'actor_rollout_ref.rollout.logprobs_mode':'processed_logprobs',
        'actor_rollout_ref.rollout.agent.num_workers':cfg['reward_workers'],
        'actor_rollout_ref.ref.log_prob_use_dynamic_bsz':True,
        'algorithm.adv_estimator':'grpo','algorithm.use_kl_in_reward':False,
        'algorithm.norm_adv_by_std_in_grpo':True,'algorithm.filter_groups.enable':False,
        'algorithm.rollout_correction.bypass_mode':False,
        'algorithm.rollout_correction.rollout_is':None,'algorithm.rollout_correction.rollout_rs':None,
        'reward.num_workers':cfg['reward_workers'],'reward.reward_model.enable':False,
        'reward.custom_reward_function.path':str(Path(__file__).with_name('verl_pilot_reward.py').resolve()),
        'reward.custom_reward_function.name':'compute_score',
        'trainer.nnodes':1,'trainer.n_gpus_per_node':1,'trainer.logger':['console'],
        'trainer.project_name':'posttrain-engineering-pilot','trainer.experiment_name':output.name,
        'trainer.total_epochs':cfg['total_steps'],'trainer.total_training_steps':cfg['total_steps'],
        'trainer.save_freq':cfg['save_freq'],'trainer.max_actor_ckpt_to_keep':cfg['checkpoint_keep'],
        'trainer.default_local_dir':str((output/'checkpoints').resolve()),
        'trainer.val_before_train':False,'trainer.test_freq':-1,
        'trainer.resume_mode':'resume_path' if resume else 'disable',
        'trainer.resume_from_path':str(resume.resolve()) if resume else None,
        'transfer_queue.enable':False,
    }
    for key,value in values.items():OmegaConf.update(config,key,value,merge=False)
    OmegaConf.update(config,'reward.custom_reward_function.reward_kwargs',
        {'config_path':str(reward_path.resolve()),'config_sha256':reward_sha256},force_add=True)
    OmegaConf.resolve(config)
    return config


def run_trainer(config,settings,trace_identity,output,resume_receipt):
    """Imports GPU dependencies only after caller verifies all launch inputs."""
    import ray
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.utils import create_rl_dataset,create_rl_sampler,need_critic,need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass,validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.config import HFModelConfig
    from posttrain.training.verl_pilot_hooks import instrumented_trainer
    from posttrain.training.verl_pilot_trace import PilotTrace

    @ray.remote
    class PilotTaskRunner(BaseTaskRunner):
        def run(self,config):
            actor,group=self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
            self.add_ref_policy_worker(config,actor)
            validate_config(config=config,use_reference_policy=need_reference_policy(config),use_critic=need_critic(config))
            model=omega_conf_to_dataclass(config.actor_rollout_ref.model,dataclass_type=HFModelConfig)
            tokenizer=model.tokenizer;processor=model.processor
            train=create_rl_dataset(config.data.train_files,config.data,tokenizer,processor,is_train=True)
            dev=create_rl_dataset(config.data.val_files,config.data,tokenizer,processor,is_train=False)
            with PilotTrace(output/'audit',trace_identity,settings['token_cap'],resume=bool(resume_receipt)) as trace:
                trainer=instrumented_trainer(RayPPOTrainer)(config=config,tokenizer=tokenizer,processor=processor,
                    role_worker_mapping=self.role_worker_mapping,resource_pool_manager=self.init_resource_pool_mgr(config),
                    ray_worker_group_cls=group,train_dataset=train,val_dataset=dev,collate_fn=collate_fn,
                    train_sampler=create_rl_sampler(config.data,train),pilot_trace=trace,
                    pilot_settings={**settings,'resume_receipt':str(resume_receipt) if resume_receipt else None})
                trainer.init_workers();trainer.fit()
    run_ppo(config,PilotTaskRunner)


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','verl-source','model-snapshot','dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('preview','reward-context','model-files','gpu-probe','attestation','gateway-identity','resume-receipt'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();cfg=checked_config(json.loads(args.config.read_text(encoding='utf-8')))
    verified=verify_source(args.verl_source)
    reward_path=args.reward_context or args.output/'reward-context-not-created.json'
    reward_hash=file_digest(reward_path) if reward_path.is_file() else None
    resume=None
    if args.resume_receipt:
        resume=Path(json.loads(args.resume_receipt.read_text(encoding='utf-8'))['checkpoint'])
    config=compose_config(cfg,args.verl_source,args.model_snapshot,args.dataset,args.output,reward_path,reward_hash,resume)
    from omegaconf import OmegaConf
    if args.preview:
        if args.preview.exists():raise ValueError('immutable_preview_exists')
        write_json(args.preview,{'created_at':utc_now(),'configuration_only_not_launchable':True,
            'source_files_verified':verified,'model_generations':0,'model_updates':0,
            'config':OmegaConf.to_container(config,resolve=True)})
        print('Configuration composed; no GPU or model process started.');return
    if sys.platform!='linux':raise RuntimeError('Linux_CUDA_required')
    if any(getattr(args,k) is None for k in ('reward_context','model_files','gpu_probe','attestation','gateway_identity')):
        raise ValueError('model_GPU_and_scoring_evidence_required_before_launch')
    from posttrain.training.verl_pilot_reward import load_context
    context,prompts,client=load_context(str(reward_path),reward_hash)
    data=checked_dataset(args.dataset,context['prompt_manifests'])
    if data['train_prompts']<cfg['batch_prompts']:raise ValueError('pilot_batch_larger_than_training_set')
    model_manifest=json.loads(args.model_files.read_text(encoding='utf-8'))
    if model_manifest!=json.loads((ROOT/'manifests/qwen7b_model_files.json').read_text(encoding='utf-8')):
        raise ValueError('official_fixed_revision_model_manifest_required')
    checked_model_files(args.model_snapshot,model_manifest)
    probe=json.loads(args.gpu_probe.read_text(encoding='utf-8'))
    if probe.get('passed') is not True or probe.get('bf16_kernel_and_backward') is not True:
        raise ValueError('real_GPU_probe_required')
    import torch
    import verl
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()
        or Path(verl.__file__).resolve().parent!=(args.verl_source/'verl').resolve()):
        raise ValueError('verified_single_GPU_and_pinned_verl_import_required')
    from posttrain.execution.contracts import Limits
    from posttrain.execution.sandboxfusion import validate_attestation
    record=json.loads(args.attestation.read_text(encoding='utf-8'))
    validate_attestation(args.attestation,record['endpoint'],Limits(time_s=30))
    expected_version='sandboxfusion-docker:'+record['image_digest']+':'+record['deployment_id']
    expected_gateway=json.loads(args.gateway_identity.read_text(encoding='utf-8'))
    if context['executor_version']!=expected_version or expected_gateway['executor_version']!=expected_version:
        raise ValueError('scoring_attestation_identity_mismatch')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(client.endpoint+'/identity',timeout=15) as response:live_gateway=json.load(response)
    if live_gateway!=expected_gateway:raise ValueError('live_scoring_gateway_changed')
    from transformers import AutoTokenizer
    from posttrain.training.sft_data import chat_tokens
    tokenizer=AutoTokenizer.from_pretrained(args.model_snapshot,local_files_only=True,trust_remote_code=False)
    if any(len(chat_tokens(tokenizer,p['messages'],add_generation_prompt=True))>cfg['max_prompt_tokens'] for p in prompts.values()):
        raise ValueError('pilot_prompt_too_long_no_silent_filtering')
    artifacts={'dataset':file_digest(args.dataset/'manifest.json'),'reward':reward_hash,
        'model_files':file_digest(args.model_files),'software_lock':file_digest(ROOT/'environments/verl-0.9.1/uv.lock'),
        'source':{p.name:file_digest(p) for p in Path(__file__).parent.glob('verl_pilot*.py')}}
    identity={'scope':'engineering_pilot_only','run_uuid':context['run_uuid'],
        'config_sha256':digest({'settings':cfg,'artifacts':artifacts})}
    args.output.mkdir(parents=True,exist_ok=bool(resume))
    write_json(args.output/('resume-config.json' if resume else 'resolved-config.json'),OmegaConf.to_container(config,resolve=True))
    import fcntl
    with (args.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_trainer(config,cfg,identity,args.output,args.resume_receipt)


if __name__=='__main__':main()
