"""Opt-in, one-copy 7B Agent GRPO pilot; Linux CUDA execution only.

The existing audited verl/vLLM path remains the reference implementation.
This runner is a feasibility experiment, not a formal-training backend.
"""
from __future__ import annotations

import json
import contextlib
import random
import time
from pathlib import Path
from uuid import uuid4

from posttrain.common import digest, file_digest, read_jsonl, utc_now, write_json
from posttrain.training.single_model_rl import (
    clipped_grpo_loss, group_advantages, rollout_episode, training_reward,
)


def checked_single_model_settings(cfg):
    if cfg.get('scope') != 'engineering_pilot_only' or cfg.get('mode') != 'agent':
        raise ValueError('single_model_agent_pilot_only')
    if (cfg.get('reward_kind') != 'binary' or cfg.get('scoring_mode') != 'binary_fail_fast'
            or cfg.get('seed') != 42 or cfg.get('g') != 4 or cfg.get('batch_prompts') != 4
            or cfg.get('max_tool_calls') != 3):
        raise ValueError('single_model_fixed_comparison_settings_required')
    bonus = cfg.get('pilot_format_bonus', 0.)
    if type(bonus) not in (int, float) or bonus not in (0, .1):
        raise ValueError('single_model_pilot_format_bonus_must_be_zero_or_point_one')
    return cfg


def _logprobs(model, tokens, mask, *, with_grad):
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    selected = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    if selected.shape != ids.shape or bool(selected[:, 0].any()):
        raise ValueError('single_model_logprob_mask_alignment')
    positions = torch.nonzero(selected[0, 1:], as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('empty_policy_generation')
    context = torch.enable_grad() if with_grad else torch.no_grad()
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
    with context, autocast:
        # Qwen2/Transformers 5.9 accepts tensor positions here. Projecting only
        # predecessor positions avoids materializing the full 8192 x 152k logits.
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=positions).logits[0].float()
        targets = ids[0, 1:][positions]
        result = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    if result.numel() != sum(mask) or not bool(torch.isfinite(result).all()):
        raise ValueError('invalid_sampled_token_logprobs')
    return result


def _save_checkpoint(output, step, model, optimizer, identity):
    import torch

    target = output / f'checkpoint-{step:06d}'
    temporary = output / f'.checkpoint-{step:06d}-incomplete-{uuid4().hex}'
    if target.exists():
        raise ValueError('complete_checkpoint_already_exists')
    temporary.mkdir()
    model.save_pretrained(temporary / 'adapter', safe_serialization=True)
    state = {'step':step, 'identity_sha256':digest(identity), 'optimizer':optimizer.state_dict(),
             'python_rng':random.getstate(), 'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all()}
    torch.save(state, temporary / 'training_state.pt')
    write_json(temporary / 'manifest.json', {'step':step, 'identity_sha256':digest(identity),
        'training_state_sha256':file_digest(temporary / 'training_state.pt'),
        'adapter_files_sha256':{p.name:file_digest(p) for p in (temporary / 'adapter').iterdir() if p.is_file()}})
    temporary.rename(target)
    write_json(output / 'last.json', {'step':step, 'path':str(target),
        'checkpoint':str(target), 'complete_update':True})
    return target


def _checked_resume(path, identity):
    import torch

    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('identity_sha256') != digest(identity):
        raise ValueError('single_model_resume_identity_changed')
    if file_digest(path / 'training_state.pt') != manifest['training_state_sha256']:
        raise ValueError('single_model_resume_state_changed')
    expected_files = set(manifest['adapter_files_sha256'])
    if {p.name for p in (path / 'adapter').iterdir() if p.is_file()} != expected_files:
        raise ValueError('single_model_resume_adapter_file_set_changed')
    for name, sha in manifest['adapter_files_sha256'].items():
        if Path(name).name != name or file_digest(path / 'adapter' / name) != sha:
            raise ValueError('single_model_resume_adapter_changed')
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != manifest['step'] or state['identity_sha256'] != digest(identity):
        raise ValueError('single_model_resume_step_or_identity_changed')
    return state


def run(cfg, model_snapshot: Path, context_path: Path, context_hash: str,
        output: Path, identity: dict, resume_receipt: Path | None = None,
        stop_after: int | None = None):
    """Run only after code_rl has verified the GPU, model, scorer and tool."""
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from posttrain.agent.tool_service import ToolClient
    from posttrain.training.sft import parameter_hash, update_plan
    from posttrain.training.verl_pilot_reward import load_context, score_candidate

    checked_single_model_settings(cfg)
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= cfg['total_steps']):
        raise ValueError('invalid_single_model_stop_after')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError('one_real_bf16_GPU_required')
    context, prompts, scorer = load_context(str(context_path), context_hash)
    if context['scope'] != 'engineering_pilot_only' or context['reward_kind'] != 'binary':
        raise ValueError('single_model_pilot_context_required')
    rows = list(read_jsonl(next(Path(x['path']) for x in context['prompt_manifests']
                                if Path(x['path']).name == 'agent-train.jsonl')))
    if len(rows) < cfg['batch_prompts'] or any(p['problem_id'] not in prompts for p in rows):
        raise ValueError('single_model_train_prompt_manifest_required')
    plan = update_plan(len(rows), cfg['batch_prompts'], cfg['total_steps'], cfg['seed'])
    if resume_receipt:
        checkpoint = Path(json.loads(resume_receipt.read_text(encoding='utf-8'))['checkpoint'])
        state = _checked_resume(checkpoint, identity)
        start_step = state['step']
        if start_step >= cfg['total_steps']:
            raise ValueError('single_model_no_remaining_updates')
    else:
        checkpoint, state, start_step = None, None, 0
        write_json(output / 'single-model-identity.json', identity)
    if json.loads((output / 'single-model-identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('single_model_output_identity_changed')
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('single_model_tokenizer_eos_or_pad_missing')
    base = AutoModelForCausalLM.from_pretrained(model_snapshot, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', trust_remote_code=False).to('cuda')
    if sum(p.numel() for p in base.parameters()) < 6_000_000_000:
        raise ValueError('not_the_7b_model')
    if checkpoint:
        model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=32,
            lora_alpha=64, lora_dropout=0., target_modules='all-linear', bias='none'))
    if not hasattr(model, 'disable_adapter'):
        raise ValueError('peft_adapter_disabled_reference_required')
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(p.requires_grad and 'lora_' not in name for name, p in model.named_parameters()):
        raise ValueError('unexpected_trainable_base_parameter')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    base_before = parameter_hash(model, False)
    lora_before = parameter_hash(model, True)
    tool = ToolClient(context['tool_endpoint'], context['executor_version'])
    session_id = uuid4().hex
    last_checkpoint = checkpoint
    session_nonzero_advantages = 0
    session_binary_successes = 0
    session_format_acceptances = 0
    end_step = cfg['total_steps'] if stop_after is None else stop_after
    if end_step <= start_step:
        raise ValueError('single_model_no_updates_before_stop')
    for step_index in range(start_step, end_step):
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        batch = []
        for row_index in plan[step_index]:
            prompt = rows[row_index]
            group = []
            for _ in range(cfg['g']):
                model.eval()
                def generate(ids, limit):
                    with torch.no_grad():
                        source = torch.tensor([ids], dtype=torch.long, device='cuda')
                        result = model.generate(input_ids=source, attention_mask=torch.ones_like(source),
                            max_new_tokens=limit, do_sample=True, temperature=1., top_p=1., top_k=0,
                            repetition_penalty=1., use_cache=True, pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    return result[0, len(ids):].tolist()
                trace = rollout_episode(prompt['messages'], tokenizer, generate, tool,
                    max_prompt_tokens=cfg['max_prompt_tokens'], max_response_tokens=cfg['max_response_tokens'],
                    max_generated_tokens=cfg['max_generated_tokens'], max_turn_tokens=cfg['max_turn_tokens'],
                    max_tool_calls=cfg['max_tool_calls'])
                request_id = uuid4().hex
                code = trace['final_code'] or 'def <POSTTRAIN_FORMAT_REJECTED>(\n'
                score = score_candidate(code, prompt, scorer, output / 'scoring', request_id,
                    context['reward_kind'], context_hash)
                reward_components = training_reward(trace, score, cfg.get('pilot_format_bonus', 0.))
                trace.update(problem_id=prompt['problem_id'], request_id=request_id,
                    reward=reward_components['training_reward'], **reward_components,
                    score_receipt=score, step=step_index + 1)
                session_binary_successes += reward_components['binary_score'] == 1.
                session_format_acceptances += not score['format_rejected']
                write_json(output / 'episodes' / f'{request_id}.json', trace)
                group.append(trace)
            advantages = group_advantages([x['reward'] for x in group])
            for trace, advantage in zip(group, advantages):
                trace['advantage'] = advantage
                batch.append(trace)
                session_nonzero_advantages += advantage != 0.
        generation_and_scoring_s = time.monotonic() - started
        model.eval()
        for trace in batch:
            trace['old_logprobs'] = _logprobs(model, trace['tokens'], trace['policy_mask'], with_grad=False).cpu()
            with model.disable_adapter():
                trace['ref_logprobs'] = _logprobs(model, trace['tokens'], trace['policy_mask'], with_grad=False).cpu()
            if trace['old_logprobs'].numel() == 0:
                raise ValueError('empty_policy_generation')
            write_json(output / 'episodes' / f"{trace['request_id']}.json",
                {k:v for k,v in trace.items() if k not in ('old_logprobs','ref_logprobs')}
                | {'old_logprobs':trace['old_logprobs'].tolist(),
                   'ref_logprobs':trace['ref_logprobs'].tolist(),
                   'policy_version':step_index})
        update_started = time.monotonic()
        token_total = sum(sum(x['policy_mask']) for x in batch)
        if token_total < 1:
            raise ValueError('empty_policy_update')
        # Old/ref probabilities stay fixed across both updates of this rollout batch.
        epoch_losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss_sum = 0.
            for trace in batch:
                current = _logprobs(model, trace['tokens'], trace['policy_mask'], with_grad=True)
                loss = clipped_grpo_loss(current, trace['old_logprobs'].to('cuda'),
                    trace['ref_logprobs'].to('cuda'), advantage=trace['advantage'],
                    clip_ratio=.2, kl_coef=.001)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite_single_model_grpo_loss')
                weight = current.numel() / token_total
                (loss * weight).backward()
                loss_sum += float(loss.detach()) * weight
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(loss_sum)
        torch.cuda.synchronize()
        step = step_index + 1
        step_report = {'step':step, 'evidence_level':'single_model_7b_agent_rl_pilot',
            'model_copies':1, 'sample_count':len(batch), 'generated_tokens':token_total,
            'rewards':[x['reward'] for x in batch], 'advantages':[x['advantage'] for x in batch],
            'binary_scores':[x['binary_score'] for x in batch],
            'format_bonuses':[x['format_bonus_applied'] for x in batch],
            'reward_protocol':batch[0]['reward_protocol'],
            'epoch_losses':epoch_losses, 'gradient_norm':float(norm),
            'generation_and_scoring_s':generation_and_scoring_s,
            'update_s':time.monotonic() - update_started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'device':torch.cuda.get_device_name(), 'completed_at':utc_now()}
        write_json(output / 'steps' / session_id / f'{step:06d}.json', step_report)
        # A rollout can fail after several completed updates. Keep every
        # completed pilot update restartable instead of replaying paid GPU work.
        last_checkpoint = _save_checkpoint(output, step, model, optimizer, identity)
    report = {'evidence_level':'single_model_7b_agent_rl_pilot', 'complete':end_step==cfg['total_steps'],
        'complete_updates':end_step, 'session_updates':end_step-start_step,
        'model_copies':1, 'gpu_verified':True, 'base_unchanged':base_before==parameter_hash(model, False),
        'lora_changed':lora_before!=parameter_hash(model, True),
        'session_nonzero_advantages':session_nonzero_advantages,
        'session_binary_successes':session_binary_successes,
        'session_format_acceptances':session_format_acceptances,
        'reward_protocol':'binary_plus_format_bonus_v1' if cfg.get('pilot_format_bonus', 0.) else 'binary_v1',
        'last_checkpoint':str(last_checkpoint), 'finished_at':utc_now(),
        'formal_training_eligible':False}
    if not report['base_unchanged'] or (session_nonzero_advantages and not report['lora_changed']):
        raise RuntimeError('single_model_parameter_update_acceptance_failed')
    write_json(output / 'single-model-result.json', report)
    return report
