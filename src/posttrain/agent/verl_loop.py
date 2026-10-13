"""Multi-turn adapter for the locally pinned verl AgentLoopBase API.

Imported only inside the Linux GPU runtime. Context tokens get mask zero through
verl's continuous-token builder; only assistant-generated tokens are optimized.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from posttrain.agent.episode import Episode, tool_observation_message, truncate_at_first_action
from posttrain.agent.tool_service import ToolClient
from posttrain.common import digest, write_json
from posttrain.training.verl_pilot_reward import load_context, score_candidate


class CodeAgentLoop(AgentLoopBase):
    def __init__(self, *args, context_path, context_sha256, max_tool_calls=3,
                 max_generated_tokens=4096, max_turn_tokens=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.context, self.prompts, self.scorer = load_context(context_path, context_sha256)
        self.context_sha256 = context_sha256
        self.tool = ToolClient(self.context['tool_endpoint'], self.context['executor_version'])
        self.max_tool_calls = max_tool_calls
        self.max_generated_tokens = max_generated_tokens
        self.max_turn_tokens = max_turn_tokens

    def _cap_text_prompt_length(self, prompt_ids):
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError('agent_prompt_overflow_no_truncation')
        return prompt_ids

    async def run(self, sampling_params, priority=0, **kwargs):
        messages = [dict(m) for m in kwargs['raw_prompt']]
        prompt = self.prompts[kwargs['extra_info']['problem_id']]
        if messages != prompt['messages']:
            raise ValueError('agent_prompt_changed')
        tokens = await self.ct_build_initial_tokens(messages)
        initial = list(tokens)
        mask, logprobs = [], []
        episode = Episode(max_tool_calls=self.max_tool_calls, max_turns=self.max_tool_calls+1)
        rid = uuid4().hex
        audit = Path(self.context['audit_directory'])/'episodes'/f'{rid}.json'
        trajectory = {'request_id': rid, 'problem_id': prompt['problem_id'], 'state':'started',
                      'context_sha256': self.context_sha256, 'turns':[]}
        write_json(audit, trajectory)
        generated, generation_s, tool_s = 0, 0., 0.
        while not episode.end_reason:
            allowance = min(self.max_turn_tokens, self.max_generated_tokens-generated,
                            self.rollout_config.response_length-len(mask))
            if allowance <= 0:
                episode.exhaust()
                break
            params = dict(sampling_params)
            params['max_tokens'] = allowance
            start = time.monotonic()
            output = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=tokens,
                                                       sampling_params=params, priority=int(priority))
            generation_s += time.monotonic()-start
            if not output.token_ids or len(output.token_ids)>allowance or output.log_probs is None or len(output.log_probs)!=len(output.token_ids):
                raise ValueError('missing_or_invalid_agent_generation_logprobs')
            # v5b stop rule: drop tokens (and their log-probs) after the first complete action.
            turn_ids = truncate_at_first_action(list(output.token_ids), lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True))
            turn_logprobs = list(output.log_probs)[:len(turn_ids)]
            generated += len(turn_ids)
            merged, mask, logprobs = await self.ct_merge_assistant_token(
                tokens, turn_ids, mask, logprobs, assistant_logprobs=turn_logprobs)
            tokens = merged.token_ids
            text = self.tokenizer.decode(turn_ids, skip_special_tokens=True)
            messages.append({'role':'assistant', 'content':text})
            trajectory['turns'].append({'token_ids': turn_ids, 'log_probs': turn_logprobs,
                                        'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
            action = episode.accept(text)
            trajectory.update(generated_tokens=generated, state='in_progress')
            write_json(audit, trajectory)
            if action['action'] != 'run_python':
                break
            if generated >= self.max_generated_tokens:
                episode.exhaust()
                break
            start = time.monotonic()
            observation = await asyncio.to_thread(self.tool.run, action['code'], action['stdin'], uuid4().hex)
            tool_s += time.monotonic()-start
            previous = list(messages)
            # This user-role message is environment context, never policy output.
            messages.append(tool_observation_message(observation, episode.max_tool_calls-episode.tool_calls))
            merged, next_mask, next_logprobs = await self.ct_merge_context_msg(previous, messages, tokens, mask, logprobs)
            if len(next_mask) >= self.rollout_config.response_length:
                episode.exhaust()
                break  # Do not append/truncate a partial observation.
            tokens, mask, logprobs = merged.token_ids, next_mask, next_logprobs
        if tokens[:len(initial)] != initial or len(tokens)-len(initial) != len(mask):
            raise ValueError('agent_prefix_or_mask_alignment_changed')
        if sum(mask) != generated or len(mask) != len(logprobs):
            raise ValueError('assistant_mask_must_cover_exact_generated_tokens')
        # A missing/invalid submission is an attributable policy failure, not infra.
        code = episode.final_code or 'def <POSTTRAIN_FORMAT_REJECTED>(\n'
        start = time.monotonic()
        score = await asyncio.to_thread(score_candidate, code, prompt, self.scorer,
                                        Path(self.context['audit_directory']), rid,
                                        self.context['reward_kind'], self.context_sha256)
        score_s = time.monotonic()-start
        trajectory.update(state='complete', end_reason=episode.end_reason, tool_calls=episode.tool_calls,
                          action_events=episode.events,
                          submission_format=episode.submission_format,
                          response_ids=tokens[len(initial):], response_mask=mask,
                          final_code_sha256=digest(code), reward=score['score'])
        write_json(audit, trajectory)
        return AgentLoopOutput(prompt_ids=initial, response_ids=tokens[len(initial):],
                               response_mask=mask, response_logprobs=logprobs,
                               reward_score=score['score'], num_turns=len(messages),
                               metrics={'generate_sequences':generation_s,'tool_calls':tool_s,'compute_score':score_s},
                               extra_fields={'reward_extra_info': {**score, 'tool_calls':episode.tool_calls,
                                             'end_reason':episode.end_reason,
                                             'submission_format':episode.submission_format,
                                             'trajectory_sha256':digest(trajectory)}})
