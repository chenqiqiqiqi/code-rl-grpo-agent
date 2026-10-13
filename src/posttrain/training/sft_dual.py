"""Independent S/A experimental SFT. Requires frozen admission and engineering gates.

No candidate program is executed here. Development greedy scoring is a separate
generation -> isolated scorer job over every preregistered saved adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import signal
import sys
import time
import uuid
from pathlib import Path

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json
from posttrain.data.schema import Problem
from posttrain.training.sft_input import load_example
from posttrain.training.sft_data import (data_length_report,encode_examples,microbatch_loss_weights,
    microbatch_token_counts,pad_batch,partition_micro_batches,resolve_pad_token_id)

from posttrain.training.sft_dual_contract import validate_routes,route_plan
from posttrain.training.sft_dual_objective import backward_routes
from posttrain.training import sft_kl_objective
from posttrain.training.sft_kl_objective import checked_objective,teacher_log_probs,masked_forward_kl

MODEL='Qwen/Qwen2.5-Coder-7B-Instruct'


def checked_config(cfg):
    if cfg.get('model_id')!=MODEL or not re.fullmatch('[a-f0-9]{40}',cfg.get('model_revision','')):
        raise ValueError('fixed_qwen_7b_revision_required')
    if cfg.get('dtype')!='bfloat16' or cfg.get('quantization'):
        raise ValueError('protocol_requires_bf16_lora')
    for key in ('max_steps','batch_size','max_length','lora_rank','lora_alpha'):
        if type(cfg.get(key)) is not int or cfg[key]<1:
            raise ValueError('invalid_'+key)
    micro=cfg.setdefault('micro_batch_size',1)
    if type(micro) is not int or not 1<=micro<=cfg['batch_size']:
        raise ValueError('invalid_micro_batch_size')
    saves=cfg.get('save_steps',[])
    if (not saves or sorted(set(saves))!=saves or saves[-1]!=cfg['max_steps']
        or any(type(x) is not int or x<1 for x in saves)):
        raise ValueError('preregister_complete_checkpoint_schedule')
    if type(cfg.get('seed')) is not int or not 0<float(cfg.get('learning_rate',0))<1:
        raise ValueError('invalid_seed_or_learning_rate')
    
    if cfg.get('objective')!='separate_s_a_v1' or cfg.get('arm') not in ('C','P','B'):raise ValueError('explicit_dual_route_required')
    if type(cfg.get('kl_beta')) not in (int,float) or cfg['kl_beta']!=(0 if cfg['arm']=='C' else 1):raise ValueError('fixed_arm_beta_required')
    return cfg


def checked_examples(train,dev):
    if not train or not dev or any(p.validation_status!='reference_passed' or not p.cluster_id for p in train+dev):
        raise ValueError('reference_validated_clustered_train_and_dev_required')
    if {p.cluster_id for p in train}&{p.cluster_id for p in dev}:
        raise ValueError('train_dev_cluster_leak')
    if len({p.problem_id for p in train+dev})!=len(train)+len(dev):
        raise ValueError('duplicate_train_dev_id')


def update_plan(size,batch_size,steps,seed):
    if min(size,batch_size,steps)<1:raise ValueError('empty_update_plan')
    rng=random.Random(seed);stream=[]
    while len(stream)<batch_size*steps:
        epoch=list(range(size));rng.shuffle(epoch);stream.extend(epoch)
    return [stream[i*batch_size:(i+1)*batch_size] for i in range(steps)]


def parameter_hash(model,trainable):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad!=trainable:continue
        h.update(f'{name}|{p.dtype}|{tuple(p.shape)}'.encode())
        flat=p.detach().reshape(-1)
        for start in range(0,p.numel(),4*1024*1024):
            chunk=flat[start:start+4*1024*1024].contiguous().view(torch.uint8).cpu().numpy()
            h.update(chunk.tobytes())
    return h.hexdigest()


def checked_checkpoint(path):
    manifest=json.loads((path/'state_manifest.json').read_text(encoding='utf-8'))
    if manifest['state_sha256']!=file_digest(path/'training_state.pt'):
        raise ValueError('checkpoint_state_checksum_mismatch')
    actual={p.name:file_digest(p) for p in (path/'adapter').iterdir() if p.is_file()}
    if actual!=manifest['adapter_sha256']:
        raise ValueError('adapter_checksum_mismatch')
    return manifest


def input_identity(examples,software):
    from posttrain.data import schema
    from posttrain.training import sft_data,sft_input
    from posttrain import common
    return {'encoded_examples_sha256':digest(examples),'software':software,
        'source_sha256':{m.__name__:file_digest(Path(m.__file__)) for m in (schema,sft_data,sft_input,common,sft_kl_objective)},
        'sft_source_sha256':file_digest(Path(__file__))}


def require_latest_checkpoint(output,resume):
    checkpoints=sorted((p for p in output.glob('checkpoint-*')
        if p.is_dir() and re.fullmatch('checkpoint-[0-9]{6}',p.name)),key=lambda p:int(p.name[11:]))
    if not checkpoints or checkpoints[-1].resolve()!=resume.resolve():
        raise ValueError('resume_latest_complete_checkpoint_not_stale_output_tail')


def train(config_path,train_path,dev_path,output,*,resume=None,stop_after=None,model_snapshot=None):
    cfg=checked_config(json.loads(config_path.read_text(encoding='utf-8')))
    if stop_after is not None and not 0<stop_after<=cfg['max_steps']:raise ValueError('invalid_stop_after')
    if resume:require_latest_checkpoint(output,resume)
    bundle=json.loads(train_path.read_text(encoding='utf-8'))
    if bundle.get('schema')!='admitted_s32_a16_v1' or file_digest(train_path)!=cfg['bundle_sha256']:raise ValueError('frozen_admission_bundle_required')
    selected_s=bundle['S'];selected_a=bundle['P' if cfg['arm']=='C' else cfg['arm']]
    validate_routes(selected_s,selected_a,bundle['heldout_clusters'])
    if sys.platform!='linux':raise RuntimeError('GPU_training_requires_Linux')
    import torch
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('real_bf16_CUDA_device_required')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    model_source=MODEL;load_options={'revision':cfg['model_revision']}
    if model_snapshot is not None:
        from posttrain.training.verl_pilot import checked_model_files,ROOT
        checked_model_files(model_snapshot,json.loads((ROOT/'manifests/qwen7b_model_files.json').read_text(encoding='utf-8')))
        model_source=str(model_snapshot);load_options={'local_files_only':True}
    tokenizer=AutoTokenizer.from_pretrained(model_source,**load_options,trust_remote_code=False)
    pad_token_id,pad_token_source=resolve_pad_token_id(tokenizer)
    examples=[(r['problem_id'],r['sample']) for r in selected_s];rejected=[]
    samples=[r['sample'] for r in selected_s];samples_a=[r['sample'] for r in selected_a]
    order,a_order=route_plan(len(samples),len(samples_a),cfg['batch_size'],cfg['max_steps'],cfg['seed'])
    identity={'config':cfg,'train_sha256':file_digest(train_path),'dev_sha256':file_digest(dev_path),
              'accepted_ids':[x[0] for x in examples],
              'length_rejected_ids':[r['problem_id'] for r in rejected if r['reason']=='example_exceeds_context_limit'],
              'target_rejected_ids':[r['problem_id'] for r in rejected if r['reason']=='empty_shifted_target'],
              'preservation_order_sha256':digest(a_order),'preservation_ids':[r['problem_id'] for r in selected_a],'preservation_samples_sha256':digest(samples_a),'dual_source_sha256':file_digest(Path(backward_routes.__code__.co_filename)),'contract_source_sha256':file_digest(Path(validate_routes.__code__.co_filename)),'update_order_sha256':digest(order),'source_sha256':file_digest(Path(__file__)),
              'tokenizer_chat_template_sha256':digest(tokenizer.chat_template),
              'tokenizer_pad_token_id':pad_token_id,'tokenizer_pad_token_source':pad_token_source,
              'input_identity':input_identity(examples,{k:importlib.metadata.version(k)
                  for k in ('torch','transformers','peft','tokenizers','safetensors','accelerate')})}
    output.mkdir(parents=True,exist_ok=resume is not None)
    if resume:
        if json.loads((output/'identity.json').read_text(encoding='utf-8'))!=identity:
            raise ValueError('resume_identity_changed')
        state_manifest=checked_checkpoint(resume)
        if state_manifest['step']>=cfg['max_steps'] or (stop_after and stop_after<=state_manifest['step']):
            raise ValueError('no_remaining_updates_in_requested_resume')
    else:
        write_json(output/'identity.json',identity)
    write_json(output/'data_length_report.json',data_length_report(cfg,samples,rejected,pad_token_source))
    base=AutoModelForCausalLM.from_pretrained(model_source,**load_options,dtype=torch.bfloat16,
        attn_implementation='sdpa',trust_remote_code=False).to('cuda')
    if sum(p.numel() for p in base.parameters())<6_000_000_000:raise ValueError('not_the_7b_model')
    if resume:
        model=PeftModel.from_pretrained(base,str(resume/'adapter'),is_trainable=True)
    else:
        model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg['lora_rank'],
            lora_alpha=cfg['lora_alpha'],lora_dropout=0.0,target_modules='all-linear',bias='none'))
    trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any('lora_' not in n for n in trainable_names):
        raise ValueError('unexpected_trainable_base_parameter')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=0.0)
    warmup=max(1,int(cfg['max_steps']*.05))
    def lr_factor(step):
        return min((step+1)/warmup,max(0,(cfg['max_steps']-step)/max(1,cfg['max_steps']-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    step=0
    if resume:
        state=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=True)
        if state['trainable_names']!=trainable_names or state['identity_sha256']!=digest(identity):
            raise ValueError('checkpoint_training_identity_mismatch')
        if state['step']!=state_manifest['step']:raise ValueError('checkpoint_step_mismatch')
        if state['S_cursor']!=state['step']*cfg['batch_size'] or state['A_cursor']!=state['step']*len(selected_a):raise ValueError('dual_cursor_mismatch')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);step=state['step']
    before_base=parameter_hash(model,False);before_lora=parameter_hash(model,True)
    stop_requested=False
    def interrupt(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    checkpoint=None;session_start=time.monotonic();starting_step=step
    session_id=uuid.uuid4().hex
    write_json(output/'sessions'/(session_id+'.json'),{'started_at':utc_now(),'starting_step':step,'complete':False})
    def save(at_step):
        checkpoint=output/f'checkpoint-{at_step:06d}'
        temporary=output/f'.checkpoint-{at_step:06d}-incomplete-{uuid.uuid4().hex}'
        if checkpoint.exists():raise ValueError('complete_checkpoint_already_exists_resume_latest')
        temporary.mkdir(exist_ok=False)
        model.save_pretrained(temporary/'adapter',safe_serialization=True)
        tokenizer.save_pretrained(temporary/'adapter')
        state={'step':at_step,'identity_sha256':digest(identity),'trainable_names':trainable_names,
            'S_cursor':at_step*cfg['batch_size'],'A_cursor':at_step*len(selected_a),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(state,temporary/'training_state.pt')
        write_json(temporary/'state_manifest.json',{'step':at_step,'state_sha256':file_digest(temporary/'training_state.pt'),
            'training_identity_sha256':digest(identity),
            'adapter_sha256':{p.name:file_digest(p) for p in (temporary/'adapter').iterdir() if p.is_file()}})
        temporary.rename(checkpoint)
        write_json(output/'last.json',{'step':at_step,'path':str(checkpoint),'complete_update':True})
        return checkpoint
    if resume is None:save(0)
    model.train()
    session_peak_allocated=torch.cuda.max_memory_allocated();session_peak_reserved=torch.cuda.max_memory_reserved()
    session_nonpadding_tokens=0;session_target_tokens=0
    for indices in order[step:]:
        started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        lr_used=optimizer.param_groups[0]['lr']
        dual=backward_routes(model,samples,samples_a,indices,a_order[step],cfg['micro_batch_size'],pad_token_id,cfg['kl_beta'],'cuda')
        counts=dual['S'];tokens=counts['target_tokens']
        ce_total=dual['ce_loss'];kl_total=dual['forward_kl'];weighted_loss=dual['loss']
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step();scheduler.step();torch.cuda.synchronize();step+=1
        update_wall_s=time.monotonic()-started
        step_peak_allocated=torch.cuda.max_memory_allocated();step_peak_reserved=torch.cuda.max_memory_reserved()
        session_peak_allocated=max(session_peak_allocated,step_peak_allocated)
        session_peak_reserved=max(session_peak_reserved,step_peak_reserved)
        session_nonpadding_tokens+=counts['nonpadding_tokens'];session_target_tokens+=tokens
        write_json(output/'steps'/session_id/f'{step:06d}.json',{'step':step,'loss':float(weighted_loss),'ce_loss':ce_total,'forward_kl':kl_total,'kl_beta':cfg['kl_beta'],'objective':cfg['objective'],
            'lr_used':lr_used,'lr_next':scheduler.get_last_lr()[0],'preservation_problem_ids':[selected_a[i]['problem_id'] for i in a_order[step-1]],'preservation_counts':dual['A'],'target_tokens':tokens,'nonpadding_tokens':counts['nonpadding_tokens'],
            'padding_tokens':counts['padding_tokens'],'padding_fraction':counts['padding_fraction'],
            'processed_tokens':counts['processed_tokens'],'micro_batch_count':counts['micro_batch_count'],
            'micro_batch_sizes':[record['size'] for record in counts['micro_batches']],
            'problem_ids':[examples[i][0] for i in indices],'gradient_norm':float(norm),
            'update_wall_s':update_wall_s,'update_tokens_per_s':counts['nonpadding_tokens']/max(update_wall_s,1e-9),
            'update_target_tokens_per_s':tokens/max(update_wall_s,1e-9),
            'peak_allocated_bytes':step_peak_allocated,'peak_reserved_bytes':step_peak_reserved,
            'learning_rate':scheduler.get_last_lr()[0]})
        ending=stop_requested or (stop_after is not None and step>=stop_after) or step==cfg['max_steps']
        if step in cfg['save_steps'] or ending:checkpoint=save(step)
        if ending:break
    after_base=parameter_hash(model,False);after_lora=parameter_hash(model,True)
    session_wall_s=time.monotonic()-session_start
    report={'evidence_level':'real_7b_sft','complete':step==cfg['max_steps'],'model_id':MODEL,
        'starting_step':starting_step,'complete_updates':step,'session_updates':step-starting_step,
        'batch_size':cfg['batch_size'],'micro_batch_size':cfg['micro_batch_size'],
        'session_nonpadding_tokens':session_nonpadding_tokens,'session_target_tokens':session_target_tokens,
        'session_tokens_per_s':session_nonpadding_tokens/max(session_wall_s,1e-9),
        'base_unchanged':before_base==after_base,'lora_changed':before_lora!=after_lora,
        'parameter_hashes':{'base_before':before_base,'base_after':after_base,'lora_before':before_lora,'lora_after':after_lora},
        'last_checkpoint':str(checkpoint),'session_wall_s':session_wall_s,'finished_at':utc_now(),
        'device':torch.cuda.get_device_name(),'peak_allocated_bytes':session_peak_allocated,
        'peak_reserved_bytes':session_peak_reserved,'tokenizer_pad_token_id':pad_token_id,
        'tokenizer_pad_token_source':pad_token_source,
        'data_length_report_sha256':file_digest(output/'data_length_report.json'),
        'software':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft')},
        'development_checkpoint_selection_complete':False,'resume_loaded':resume is not None,
        'resume_equivalence_validated':False}
    write_json(output/'sessions'/(session_id+'.json'),report)
    if not report['base_unchanged'] or (step>starting_step and not report['lora_changed']):
        raise RuntimeError('parameter_update_acceptance_failed')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--model-snapshot',type=Path)
    a=p.parse_args();print(json.dumps(train(a.config,a.train,a.dev,a.output,resume=a.resume,stop_after=a.stop_after,model_snapshot=a.model_snapshot)))


if __name__=='__main__':main()
