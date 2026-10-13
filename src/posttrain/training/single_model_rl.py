"""One-copy PyTorch/PEFT Agent GRPO engineering pilot.

The rollout and actor use the *same* model object. Generation is no-grad;
sampled-token probabilities are recomputed for the policy-gradient update.
This is deliberately separate from the verified verl/vLLM backend.
"""
from __future__ import annotations

import math
import hashlib
import json
import time
from uuid import uuid4


def group_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2 or any(not math.isfinite(x) for x in rewards):
        raise ValueError('finite_reward_group_required')
    mean = sum(rewards) / len(rewards)
    variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    if variance == 0:
        return [0.] * len(rewards)
    scale = math.sqrt(variance)
    return [(x - mean) / scale for x in rewards]


def training_reward(trace: dict, score: dict, format_bonus: float) -> dict:
    """Pilot-only shaping; keep the isolated binary judgment separately auditable."""
    if type(format_bonus) not in (int, float) or format_bonus not in (0, .1):
        raise ValueError('invalid_pilot_format_bonus')
    binary = score.get('score')
    if type(binary) not in (int, float) or binary not in (0, 1):
        raise ValueError('invalid_binary_score')
    rejected = score.get('format_rejected')
    if type(rejected) is not bool:
        raise ValueError('invalid_score_format_rejection')
    submitted = trace.get('end_reason') == 'submitted'
    code = trace.get('final_code')
    has_code = isinstance(code, str) and bool(code.strip())
    source = trace.get('submission_format')
    if (submitted != has_code or (not submitted and not rejected)
            or (submitted and source not in ('json_action', 'python_fence_fallback'))
            or (not submitted and source is not None)
            or (binary == 1 and (not submitted or rejected))):
        raise ValueError('submission_score_alignment')
    applied = float(format_bonus) if submitted and not rejected and binary == 0 else 0.
    return {'binary_score':float(binary), 'format_bonus_applied':applied,
            'training_reward':float(binary) + applied,
            'reward_protocol':'binary_plus_format_bonus_v1' if format_bonus else 'binary_v1'}


def checked_trajectory(tokens: list[int], policy_mask: list[bool], generated_spans: list[tuple[int, int]]):
    if len(tokens) != len(policy_mask) or len(tokens) < 2 or policy_mask[0]:
        raise ValueError('invalid_trajectory_alignment')
    expected = [False] * len(tokens)
    for start, end in generated_spans:
        if not (0 < start < end <= len(tokens)) or any(expected[start:end]):
            raise ValueError('invalid_generated_span')
        expected[start:end] = [True] * (end - start)
    if expected != list(policy_mask):
        raise ValueError('generated_token_mask_mismatch')
    return len(tokens), sum(policy_mask)


def extend_after_tool(tokens, policy_mask, generated_spans, generated, next_prompt):
    """Append tool context while preserving every sampled ID and its mask."""
    prefix = list(tokens) + list(generated)
    if list(next_prompt[:len(prefix)]) != prefix or len(next_prompt) <= len(prefix):
        raise ValueError('tool_context_retokenized_generated_output')
    start = len(tokens)
    spans = [*generated_spans, (start, start + len(generated))]
    mask = [*policy_mask, *([True] * len(generated)), *([False] * (len(next_prompt) - len(prefix)))]
    checked_trajectory(list(next_prompt), mask, spans)
    return list(next_prompt), mask, spans


def rollout_episode(messages, tokenizer, generate_tokens, tool_client, *, max_prompt_tokens,
                    max_response_tokens, max_generated_tokens, max_turn_tokens, max_tool_calls):
    """Build an exact-ID Agent trajectory using one no-grad generation callback."""
    from posttrain.agent.episode import Episode, tool_observation_message, truncate_at_first_action
    from posttrain.training.sft_data import chat_tokens

    if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('public_two_message_prompt_required')
    messages = [dict(m) for m in messages]
    tokens = chat_tokens(tokenizer, messages, add_generation_prompt=True)
    initial_len = len(tokens)
    if initial_len > max_prompt_tokens:
        raise ValueError('agent_prompt_overflow_no_truncation')
    mask = [False] * initial_len
    spans = []
    episode = Episode(max_tool_calls=max_tool_calls, max_turns=max_tool_calls + 1)
    generated_count = 0
    times = {'generation_s': 0., 'tool_s': 0.}
    turns = []
    while not episode.end_reason:
        allowance = min(max_turn_tokens, max_generated_tokens - generated_count,
                        max_response_tokens - (len(tokens) - initial_len))
        if allowance <= 0:
            episode.exhaust()
            break
        started = time.monotonic()
        generated = list(generate_tokens(tokens, allowance))
        times['generation_s'] += time.monotonic() - started
        if not generated or len(generated) > allowance or any(type(x) is not int or x < 0 for x in generated):
            raise ValueError('invalid_agent_generation_ids')
        # v5b stop rule: tokens after the first complete action are not part of the trajectory.
        generated = truncate_at_first_action(generated, lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
        generated_count += len(generated)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        turns.append({'token_ids': generated, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
        action = episode.accept(text)
        prefix = tokens + generated
        prefix_mask = mask + [True] * len(generated)
        prefix_spans = [*spans, (len(tokens), len(prefix))]
        checked_trajectory(prefix, prefix_mask, prefix_spans)
        if action['action'] != 'run_python' or generated_count >= max_generated_tokens:
            tokens, mask, spans = prefix, prefix_mask, prefix_spans
            if action['action'] == 'run_python':
                episode.exhaust()
            break
        started = time.monotonic()
        observation = tool_client.run(action['code'], action['stdin'], uuid4().hex)
        times['tool_s'] += time.monotonic() - started
        previous = list(messages)
        messages.append({'role':'assistant', 'content':text})
        tool_message = tool_observation_message(observation, episode.max_tool_calls - episode.tool_calls)
        messages.append(tool_message)
        # A sampled sequence need not equal encode(decode(sampled_ids)). Encode
        # only the template suffix, never the assistant output being trained on.
        template_prefix = chat_tokens(tokenizer, previous, add_generation_prompt=True)
        empty_assistant = previous + [{'role':'assistant', 'content':''}, tool_message]
        template_with_tool = chat_tokens(tokenizer, empty_assistant, add_generation_prompt=True)
        if template_with_tool[:len(template_prefix)] != template_prefix:
            raise ValueError('tool_template_prefix_changed')
        suffix = template_with_tool[len(template_prefix):]
        eos = getattr(tokenizer, 'eos_token_id', None)
        if generated[-1] == eos:
            if not suffix or suffix[0] != eos:
                raise ValueError('tool_template_eos_mismatch')
            suffix = suffix[1:]
        next_prompt = tokens + generated + suffix
        if len(next_prompt) - initial_len >= max_response_tokens:
            tokens, mask, spans = prefix, prefix_mask, prefix_spans
            episode.exhaust()
            break
        tokens, mask, spans = extend_after_tool(tokens, mask, spans, generated, next_prompt)
    checked_trajectory(tokens, mask, spans)
    if sum(mask) != generated_count:
        raise ValueError('assistant_mask_must_cover_exact_generated_tokens')
    return {'tokens':tokens, 'policy_mask':mask, 'generated_spans':spans,
            'final_code':episode.final_code, 'end_reason':episode.end_reason,
            'submission_format':episode.submission_format, 'tool_calls':episode.tool_calls,
            'action_events':episode.events, 'turns':turns, **times}


def sampled_logprobs(logits, input_ids, policy_mask):
    """Return log p(token_t | tokens_<t) for exactly the generated tokens."""
    import torch

    if (logits.ndim != 3 or input_ids.ndim != 2 or policy_mask.ndim != 2
            or logits.shape[:2] != input_ids.shape or input_ids.shape != policy_mask.shape
            or bool(policy_mask[:, 0].any())):
        raise ValueError('shifted_logprob_shape_or_mask')
    # Discard context rows before converting the vocabulary dimension to FP32.
    # Long tool observations can otherwise create several GB of needless logits.
    active = policy_mask[:, 1:].bool()
    active_logits = logits[:, :-1][active].float()
    active_ids = input_ids[:, 1:][active]
    return active_logits.gather(-1, active_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(active_logits, dim=-1)


def clipped_grpo_loss(current, old, reference, *, advantage: float, clip_ratio: float, kl_coef: float):
    import torch

    if (current.shape != old.shape or current.shape != reference.shape or current.numel() == 0
            or not math.isfinite(advantage) or not 0 < clip_ratio < 1 or not 0 <= kl_coef < 1):
        raise ValueError('invalid_grpo_loss_inputs')
    ratio = torch.exp(current - old.detach())
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
    delta = reference.detach() - current
    kl = torch.exp(delta) - delta - 1
    return (-torch.minimum(unclipped, clipped) + kl_coef * kl).mean()
