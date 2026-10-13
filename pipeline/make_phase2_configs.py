"""Write the phase-2 C (single-turn) and A (Agent) configs from the chosen W (protocol v4 addenda 6/7).
Usage: python stage4-phase1/make_phase2_configs.py RUN STEP   (W = runs/stage4-phase1/RUN-export-STEP)
  e.g. kodcode2m 124 (mixed reward) or kodcode3b 60 (binary reward). Reward and the 48 GB memory settings come
  from the matching extension config single_rl_kodcode3{b|m}_v4.json (same algorithm settings as kodcode2{b|m},
  memory settings that fit adapter-initialised runs).
Both start from W's exported adapter, share reward/LR/LoRA/4x8/seed and stop on the same generated-token
budget B2 (stop_generated_tokens); total_steps is only a safety ceiling.
Outputs configs/single_rl_phase2c_v4.json and configs/agent_rl_phase2a_v4.json (refuses to overwrite).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
B2 = 1_000_000
WARMUP = 10  # protocol v4 addendum 10: linear LR warmup for the re-initialised optimizer
GPU = '/root/autodl-tmp/posttrain-project'


def write(path, cfg):
    if path.exists():
        raise SystemExit(f'{path} exists')
    path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(path)


def main():
    run, step = sys.argv[1], int(sys.argv[2])
    if run not in ('kodcode2b', 'kodcode2m', 'kodcode3b', 'kodcode3m'):
        raise SystemExit('W run must be one of the KodCode phase-1 runs')
    base = json.loads((ROOT/f'configs/single_rl_kodcode3{run[-1]}_v4.json').read_text(encoding='utf-8'))
    adapter = f'{GPU}/runs/stage4-phase1/{run}-export-{step}/adapter'
    common = {k: base[k] for k in ('scope', 'reward_kind', 'scoring_mode', 'seed', 'batch_prompts', 'g',
                                   'invalid_scoring_policy', 'learning_rate', 'kl_loss_coef', 'save_freq',
                                   'checkpoint_keep', 'save_lora_only')}
    common.update(init_lora_adapter=adapter, stop_generated_tokens=B2, lr_warmup_steps=WARMUP)
    per_step = base['batch_prompts'] * base['g']
    c = {**common, 'mode': 'single', 'max_prompt_tokens': 3072, 'max_response_tokens': 1024,
         'max_generated_tokens': 1024, 'max_turn_tokens': 1024, 'total_steps': 400,
         'gpu_memory_utilization': base['gpu_memory_utilization'], 'max_num_seqs': base['max_num_seqs'],
         'ppo_max_token_len_per_gpu': base['ppo_max_token_len_per_gpu']}
    c['token_cap'] = c['total_steps'] * per_step * c['max_response_tokens']
    a = {**common, 'mode': 'agent', 'max_prompt_tokens': 3072, 'max_response_tokens': 5120,
         'max_generated_tokens': 4096, 'max_turn_tokens': 1024, 'max_tool_calls': 3, 'total_steps': 150,
         'gpu_memory_utilization': base['gpu_memory_utilization'], 'max_num_seqs': base['max_num_seqs'],
         'ppo_max_token_len_per_gpu': 8192}
    a['token_cap'] = a['total_steps'] * per_step * a['max_response_tokens']
    sys.path.insert(0, str(ROOT/'src'))
    from posttrain.training.code_rl import checked_settings
    checked_settings(dict(c)); checked_settings(dict(a))
    write(ROOT/'configs/single_rl_phase2c_v4.json', c)
    write(ROOT/'configs/agent_rl_phase2a_v4.json', a)


if __name__ == '__main__':
    main()
