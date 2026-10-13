import math

import pytest
import torch

from posttrain.training.single_model_rl import (
    group_advantages, sampled_logprobs, clipped_grpo_loss, checked_trajectory,
    extend_after_tool,
    rollout_episode,
)
from posttrain.training import single_model_rl as reward_module
from posttrain.training.single_model_runner import checked_single_model_settings
from posttrain.training.single_model_runner import _logprobs, _save_checkpoint, _checked_resume


def test_group_advantages_zero_variance_and_centering():
    assert group_advantages([0.0, 0.0, 0.0, 0.0]) == [0.0] * 4
    values = group_advantages([0.0, 0.0, 1.0, 1.0])
    assert values == pytest.approx([-1.0, -1.0, 1.0, 1.0])


def test_pilot_format_bonus_creates_signal_without_changing_binary_score():
    assert hasattr(reward_module, 'training_reward')
    training_reward = reward_module.training_reward
    accepted = {'end_reason':'submitted', 'final_code':'print(1)',
                'submission_format':'python_fence_fallback'}
    rejected = {'end_reason':'invalid_action', 'final_code':None,
                'submission_format':None}
    wrong = training_reward(accepted, {'score':0.0, 'format_rejected':False}, .1)
    correct = training_reward(accepted, {'score':1.0, 'format_rejected':False}, .1)
    invalid = training_reward(rejected, {'score':0.0, 'format_rejected':True}, .1)
    assert wrong == {'binary_score':0.0, 'format_bonus_applied':.1,
                     'training_reward':.1, 'reward_protocol':'binary_plus_format_bonus_v1'}
    assert correct['binary_score'] == correct['training_reward'] == 1.0
    assert correct['format_bonus_applied'] == 0.0
    assert invalid['training_reward'] == invalid['format_bonus_applied'] == 0.0
    assert group_advantages([wrong['training_reward'], invalid['training_reward'],
                             invalid['training_reward'], invalid['training_reward']]) != [0.] * 4
    assert training_reward(accepted, {'score':0.0, 'format_rejected':False}, 0.)['training_reward'] == 0.


def test_pilot_format_bonus_rejects_unverified_or_inconsistent_scoring():
    training_reward = reward_module.training_reward
    submitted = {'end_reason':'submitted', 'final_code':'print(1)',
                 'submission_format':'json_action'}
    with pytest.raises(ValueError, match='score'):
        training_reward(submitted, {'score':.5, 'format_rejected':False}, .1)
    syntax_rejected = training_reward(submitted, {'score':0., 'format_rejected':True}, .1)
    assert syntax_rejected['format_bonus_applied'] == syntax_rejected['training_reward'] == 0.
    with pytest.raises(ValueError, match='alignment'):
        training_reward({'end_reason':'invalid_action', 'final_code':None,
                         'submission_format':None}, {'score':0., 'format_rejected':False}, .1)
    with pytest.raises(ValueError, match='format_bonus'):
        training_reward(submitted, {'score':0., 'format_rejected':False}, .2)


def test_sampled_logprobs_use_shifted_causal_positions_and_generated_mask():
    logits = torch.tensor([[[0., 2.], [3., 0.], [0., 4.], [2., 0.]]])
    ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[False, True, False, True]])
    selected = sampled_logprobs(logits, ids, mask)
    expected = torch.log_softmax(logits[0, [0, 2]], dim=-1)[:, 1]
    assert torch.allclose(selected, expected)


def test_clipped_grpo_loss_keeps_old_policy_fixed_and_has_gradient():
    current = torch.tensor([-0.8, -1.1], requires_grad=True)
    old = torch.tensor([-1., -1.])
    reference = torch.tensor([-1., -1.])
    loss = clipped_grpo_loss(current, old, reference, advantage=1., clip_ratio=.2, kl_coef=0.)
    loss.backward()
    assert math.isfinite(loss.item())
    assert current.grad is not None
    assert current.grad[0].item() == 0  # positive advantage beyond PPO clip
    assert current.grad[1].item() < 0


def test_trajectory_rejects_context_tokens_marked_as_generated():
    checked_trajectory([10, 11, 12], [False, True, False], [(1, 2)])
    with pytest.raises(ValueError, match='generated_token_mask_mismatch'):
        checked_trajectory([10, 11, 12], [False, True, True], [(1, 2)])


def test_tool_observation_extends_context_without_training_on_it():
    ids, mask, spans = extend_after_tool(
        [10, 11], [False, False], [], [12, 13], [10, 11, 12, 13, 14, 15]
    )
    assert ids == [10, 11, 12, 13, 14, 15]
    assert mask == [False, False, True, True, False, False]
    assert spans == [(2, 4)]
    with pytest.raises(ValueError, match='tool_context_retokenized_generated_output'):
        extend_after_tool([10, 11], [False, False], [], [12, 13], [10, 11, 12, 99])


def test_agent_rollout_uses_tool_as_context_and_only_assistant_as_policy():
    class Tokenizer:
        eos_token_id = 42

        def apply_chat_template(self, messages, **kwargs):
            return ([10, 11] if len(messages) == 2 else
                    [10, 11, 42, 14, 15] if not messages[2]['content'] else
                    [10, 11, 12, 13, 42, 14, 15])

        def decode(self, ids, **kwargs):
            return ('{"action":"run_python","code":"print(1)","stdin":""}'
                    if ids == [12, 13] else '```python\nprint(1)\n```')

    class Tool:
        def run(self, code, stdin, request_id):
            assert code == 'print(1)'
            return {'status':'ok','stdout':'1\n','stderr':'','exit_code':0,'truncated':False}

    outputs = iter([[12, 13], [16]])
    row = rollout_episode(
        [{'role':'system','content':'s'}, {'role':'user','content':'u'}], Tokenizer(),
        lambda ids, limit: next(outputs), Tool(), max_prompt_tokens=8,
        max_response_tokens=20, max_generated_tokens=5, max_turn_tokens=3,
        max_tool_calls=3,
    )
    assert row['tokens'] == [10, 11, 12, 13, 42, 14, 15, 16]
    assert row['policy_mask'] == [False, False, True, True, False, False, False, True]
    assert row['final_code'] == 'print(1)\n'
    assert row['submission_format'] == 'python_fence_fallback'


@pytest.mark.parametrize('generated_first', [[12, 13], [12, 13, 42]])
def test_agent_rollout_preserves_sampled_ids_when_tool_action_retokenizes(generated_first):
    class Tokenizer:
        eos_token_id = 42

        def apply_chat_template(self, messages, **kwargs):
            # The sampled IDs decode to a valid action, but encoding that text
            # inside the chat template chooses a different tokenization.
            return ([10, 11] if len(messages) == 2 else
                    [10, 11, 42, 14, 15] if not messages[2]['content'] else
                    [10, 11, 99, 42, 14, 15])

        def decode(self, ids, **kwargs):
            return ('{"action":"run_python","code":"print(1)","stdin":""}'
                    if ids == generated_first else '```python\nprint(1)\n```')

    class Tool:
        def run(self, code, stdin, request_id):
            assert (code, stdin) == ('print(1)', '')
            return {'status':'ok','stdout':'1\n','stderr':'','exit_code':0,'truncated':False}

    outputs = iter([generated_first, [16]])
    row = rollout_episode(
        [{'role':'system','content':'s'}, {'role':'user','content':'u'}], Tokenizer(),
        lambda ids, limit: next(outputs), Tool(), max_prompt_tokens=8,
        max_response_tokens=20, max_generated_tokens=8, max_turn_tokens=4,
        max_tool_calls=3,
    )
    expected_context = [10, 11] + generated_first + ([14, 15] if generated_first[-1] == 42 else [42, 14, 15])
    assert row['tokens'] == expected_context + [16]
    assert row['policy_mask'] == [False, False] + [True] * len(generated_first) + [False] * (len(expected_context) - 2 - len(generated_first)) + [True]
    assert row['tool_calls'] == 1


def test_single_model_engine_rejects_formal_and_changed_comparison_protocol():
    import json
    from pathlib import Path

    cfg = json.loads(Path('configs/agent_rl_pilot.json').read_text(encoding='utf-8'))
    assert checked_single_model_settings(cfg) is cfg
    shaped = json.loads(Path('configs/agent_rl_single_model_format_pilot.json').read_text(encoding='utf-8'))
    assert checked_single_model_settings(shaped) is shaped
    with pytest.raises(ValueError, match='format_bonus'):
        checked_single_model_settings({**cfg, 'pilot_format_bonus':.2})
    with pytest.raises(ValueError, match='single_model_agent_pilot_only'):
        checked_single_model_settings({**cfg, 'scope':'code_rl_v3'})
    with pytest.raises(ValueError, match='fixed_comparison'):
        checked_single_model_settings({**cfg, 'g':2})


def test_one_model_object_supports_rollout_probabilities_reference_and_lora_gradient():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_ctx=32,
                                      n_embd=16, n_layer=1, n_head=2))
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2,
                                           lora_alpha=4, target_modules=['c_attn']))
    tokens = [1, 2, 3, 4]
    mask = [False, False, True, True]
    model.eval()
    generated = model.generate(input_ids=torch.tensor([[1, 2]]), max_new_tokens=2,
                               do_sample=True, top_k=0, top_p=1., temperature=1.,
                               pad_token_id=0, eos_token_id=3)
    assert generated.shape[0] == 1
    assert generated.shape[1] > 2
    old = _logprobs(model, tokens, mask, with_grad=False)
    with torch.no_grad():
        full = model(input_ids=torch.tensor([tokens]), use_cache=False).logits
        expected_old = sampled_logprobs(full, torch.tensor([tokens]), torch.tensor([mask]))
    assert torch.allclose(old, expected_old, atol=1e-6)
    with model.disable_adapter():
        reference = _logprobs(model, tokens, mask, with_grad=False)
    model.train()
    current = _logprobs(model, tokens, mask, with_grad=True)
    loss = clipped_grpo_loss(current, old, reference, advantage=1., clip_ratio=.2,
                             kl_coef=.001)
    loss.backward()
    assert len(current) == 2
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_single_model_resume_rejects_extra_or_changed_adapter_files(tmp_path):
    class Adapter:
        def save_pretrained(self, path, **kwargs):
            path.mkdir()
            (path / 'adapter.safetensors').write_bytes(b'fixture')

    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter])
    identity = {'backend':'single_model', 'run_uuid':'fixed'}
    checkpoint = _save_checkpoint(tmp_path, 1, Adapter(), optimizer, identity)
    assert _checked_resume(checkpoint, identity)['step'] == 1
    (checkpoint / 'adapter' / 'unexpected.safetensors').write_bytes(b'extra')
    with pytest.raises(ValueError, match='adapter_file_set_changed'):
        _checked_resume(checkpoint, identity)
