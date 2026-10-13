"""Stage 4 phase 1 (protocol v4, M1 = M0): formal settings, freeze, pool data, adapter export."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from posttrain.common import file_digest
from posttrain.training import code_rl, verl_pilot
from posttrain.training.code_rl import checked_settings, check_prepared, validate_v4_freeze

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'.cache/upstream/verl-source'/('verl-' + verl_pilot.VERL_REVISION)


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def config(kind):
    return json.loads((ROOT/f'configs/single_rl_{kind}_v4.json').read_text(encoding='utf-8'))


def test_formal_configs_differ_only_in_reward():
    binary, mixed, smoke = config('binary'), config('mixed'), config('smoke')
    for cfg in (binary, mixed, smoke):
        checked_settings(cfg)
    assert {k for k in binary if binary[k] != mixed[k]} == {'reward_kind', 'scoring_mode'}
    assert {k for k in binary if binary[k] != smoke[k]} == {'token_cap', 'save_freq'}
    assert binary['max_response_tokens'] == 1024 and binary['batch_prompts'] * binary['g'] == 32


@pytest.mark.parametrize('bad', [{'learning_rate': 0}, {'kl_loss_coef': float('nan')}, {'save_freq': 0},
                                 {'checkpoint_keep': 1.5}, {'save_lora_only': 'yes'}])
def test_invalid_overrides_rejected(bad):
    with pytest.raises(ValueError):
        checked_settings({**config('binary'), **bad})


def test_compose_applies_overrides_and_keeps_pilot_defaults(tmp_path, monkeypatch):
    if not SOURCE.is_dir():
        pytest.skip('requires the pinned local verl source snapshot')
    monkeypatch.syspath_prepend(str(SOURCE.resolve()))
    formal = code_rl.compose(config('binary'), SOURCE, tmp_path/'model', tmp_path/'data', tmp_path/'f',
                             tmp_path/'context.json', 'a'*64)
    actor = formal.actor_rollout_ref.actor
    assert actor.optim.lr == 1e-5 and actor.kl_loss_coef == 0.001 and actor.checkpoint.save_lora_only is True
    assert formal.trainer.save_freq == 10 and formal.trainer.max_actor_ckpt_to_keep == 3
    assert formal.actor_rollout_ref.rollout.max_num_seqs == 16 and formal.actor_rollout_ref.rollout.n == 4
    assert formal.data.train_batch_size == 8 and formal.data.max_response_length == 1024
    pilot = code_rl.compose(json.loads((ROOT/'configs/single_rl_pilot_4090d.json').read_text()), SOURCE,
                            tmp_path/'model', tmp_path/'data', tmp_path/'p', tmp_path/'context.json', 'a'*64)
    assert pilot.actor_rollout_ref.actor.optim.lr == 1e-6 and pilot.trainer.save_freq == 10
    assert not pilot.actor_rollout_ref.actor.checkpoint.get('save_lora_only', False)


def test_phase2_runs_can_start_from_an_exported_adapter(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='absolute'):
        checked_settings({**config('binary'), 'init_lora_adapter': 'relative/adapter'})
    if not SOURCE.is_dir():
        pytest.skip('requires the pinned local verl source snapshot')
    monkeypatch.syspath_prepend(str(SOURCE.resolve()))
    cfg = {**config('binary'), 'init_lora_adapter': '/runs/w-export/adapter'}
    composed = code_rl.compose(checked_settings(cfg), SOURCE, tmp_path/'model', tmp_path/'data', tmp_path/'o',
                               tmp_path/'context.json', 'a'*64)
    assert composed.actor_rollout_ref.model.lora_adapter_path == '/runs/w-export/adapter'
    plain = code_rl.compose(config('binary'), SOURCE, tmp_path/'model', tmp_path/'data', tmp_path/'p',
                            tmp_path/'context.json', 'a'*64)
    assert plain.actor_rollout_ref.model.lora_adapter_path is None


@pytest.fixture
def frozen(tmp_path):
    """A self-contained root with the real pool/pilot report and a 3-problem prepared set."""
    root = tmp_path/'root'
    prepared = root/'data/prepared/pool'
    for rel in ('manifests/qwen7b_model_files.json', 'reports/verl-4090d-20260930/TASK4_VERL_4090D_ACCEPTANCE.md',
                'EXPERIMENT_PROTOCOL.md', 'configs/single_rl_binary_v4.json'):
        (root/rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(ROOT/rel, root/rel)
    pool = json.loads((ROOT/'data/prepared/rl-pool-v5b/rl-problem-pool.json').read_text(encoding='utf-8'))
    pool['pool'] = pool['pool'][:3]
    prepared.mkdir(parents=True)
    (prepared/'rl-problem-pool.json').write_text(json.dumps(pool), encoding='utf-8')
    (prepared/'single-train.jsonl').write_text(''.join(json.dumps({'problem_id': p['problem_id']})+'\n'
                                                      for p in reversed(pool['pool'])), encoding='utf-8')
    (prepared/'manifest.json').write_text('{}', encoding='utf-8')
    freeze = script('make_rl_freeze').build_freeze(root, 'configs/single_rl_binary_v4.json', 'data/prepared/pool', 625, 400)
    return root, prepared, freeze


def test_v4_freeze_accepts_bound_inputs_and_detects_changes(frozen):
    root, prepared, freeze = frozen
    cfg = config('binary')
    validate_v4_freeze(freeze, root, cfg, prepared)
    for change, error in (({'learning_rate': 2e-5}, 'unfrozen_run_settings'),):
        with pytest.raises(ValueError, match=error):
            validate_v4_freeze(freeze, root, {**cfg, **change}, prepared)
    for patch, error in ((lambda f: f.update(common_start='merged'), 'v4_common_start'),
                         (lambda f: f.update(total_forecast_including_spent_cny=1700), 'reserve_400'),
                         (lambda f: f.update(protocol='code_agent_rl_v3'), 'v4_freeze_required'),
                         (lambda f: f['pilot_evidence'].update(resume_passed=False), 'v4_real_pilot'),
                         (lambda f: f.update(checkpoint_rule='best_dev'), 'checkpoint_rule')):
        broken = json.loads(json.dumps(freeze)); patch(broken)
        with pytest.raises(ValueError, match=error):
            validate_v4_freeze(broken, root, cfg, prepared)
    from posttrain.common import digest
    agent = {**cfg, 'mode': 'agent'}
    shutil.copy(prepared/'single-train.jsonl', prepared/'agent-train.jsonl')
    agent_freeze = {**freeze, 'settings_sha256': digest(agent)}
    validate_v4_freeze(agent_freeze, root, agent, prepared)
    with pytest.raises(ValueError, match='real_multiturn_pilot'):
        validate_v4_freeze({**agent_freeze, 'agent_pilot_passed': False}, root, agent, prepared)


def test_v4_freeze_binds_pool_prepared_ids_and_artifacts(frozen):
    root, prepared, freeze = frozen
    cfg = config('binary')
    originals = {p: p.read_bytes() for p in (prepared/'single-train.jsonl', prepared/'rl-problem-pool.json')}
    with (prepared/'single-train.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'problem_id': 'taco-train-999999'})+'\n')
    with pytest.raises(ValueError, match='prepared_train_set_is_not_the_frozen_pool'):
        validate_v4_freeze(freeze, root, cfg, prepared)
    (prepared/'rl-problem-pool.json').write_text('{"pool": []}', encoding='utf-8')
    with pytest.raises(ValueError, match='problem_pool_changed'):
        validate_v4_freeze(freeze, root, cfg, prepared)
    for path, data in originals.items():
        path.write_bytes(data)
    validate_v4_freeze(freeze, root, cfg, prepared)
    (root/'EXPERIMENT_PROTOCOL.md').write_text('edited', encoding='utf-8')
    with pytest.raises(ValueError, match='frozen_artifact_changed'):
        validate_v4_freeze(freeze, root, cfg, prepared)


def test_make_freeze_refuses_forecast_above_1600(frozen):
    root, _, _ = frozen
    with pytest.raises(ValueError):
        script('make_rl_freeze').build_freeze(root, 'configs/single_rl_binary_v4.json', 'data/prepared/pool', 1700, 400)


def test_pool_preparation_keeps_lines_in_order_and_passes_launch_checks(tmp_path):
    prep = script('prepare_rl_pool')
    source = ROOT/'data/prepared/agent-rl-v5b'
    pool = json.loads((ROOT/'eval-baseline-20260930/rl-problem-pool.json').read_text(encoding='utf-8'))
    pool['pool'] = pool['pool'][5:8]
    pool_path = tmp_path/'pool.json'; pool_path.write_text(json.dumps(pool), encoding='utf-8')
    prep.ROOT = tmp_path  # manifest records paths relative to ROOT
    local = tmp_path/'source'; shutil.copytree(source, local)
    out = tmp_path/'out'
    prep.prepare(local, pool_path, out, ['single'])
    ids = {p['problem_id'] for p in pool['pool']}
    lines = (local/'single-train.jsonl').read_bytes().splitlines(keepends=True)
    assert (out/'single-train.jsonl').read_bytes() == b''.join(l for l in lines if json.loads(l)['problem_id'] in ids)
    assert (out/'single-development.jsonl').read_bytes() == (local/'single-development.jsonl').read_bytes()
    assert file_digest(out/'rl-problem-pool.json') == file_digest(pool_path)
    assert len(check_prepared(out, 'single')) == 2
    pool['pool'].append({'problem_id': 'taco-train-999999'}); pool_path.write_text(json.dumps(pool), encoding='utf-8')
    with pytest.raises(ValueError, match='pool_problems_missing_from_source'):
        prep.prepare(local, pool_path, tmp_path/'out2', ['single'])


@pytest.mark.parametrize('sleep_fails', [False, True])
def test_budget_end_sleeps_rollout_before_final_save(tmp_path, sleep_fails):
    from types import SimpleNamespace
    from posttrain.training.budget import BudgetExceeded
    from posttrain.training.verl_pilot_hooks import instrumented_trainer
    calls = []
    def sleep():
        calls.append('sleep')
        if sleep_fails:
            raise RuntimeError('fixture')
    trace = SimpleNamespace(root=tmp_path, session='s', ledger=SimpleNamespace(snapshot=lambda: {'remaining': 1}),
                            snapshot=lambda: {}, phase=lambda *a, **k: calls.append('phase:' + a[1]))
    class Base:
        def __init__(self):
            self.global_steps, self._last_synced_step = 5, 4
            self._last_complete_data_state = {}
            self.train_dataloader = SimpleNamespace(load_state_dict=lambda s: None)
            self.checkpoint_manager = SimpleNamespace(sleep_replicas=sleep)
            self.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)))
        def fit(self): raise BudgetExceeded('fixture')
    trainer = instrumented_trainer(Base)(pilot_trace=trace, pilot_settings={'scope': 'code_rl_v3'})
    trainer._save_checkpoint = lambda: calls.append(f'save:{trainer.global_steps}')
    trainer.fit()
    expected = ['sleep', 'phase:final_save_sleep_failed', 'save:4'] if sleep_fails else ['sleep', 'save:4']
    assert calls == expected
    assert json.loads((tmp_path/'completion.json').read_text())['last_complete_update'] == 4


MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def lora_shapes(layers=2, r=32, drop=None, bad_rank=None):
    shapes = {}
    for layer in range(layers):
        for m in MODULES:
            block = 'self_attn' if m.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj')) else 'mlp'
            base = f'base_model.model.model.layers.{layer}.{block}.{m}'
            shapes[f'{base}.lora_A.weight'] = (r, 64)
            shapes[f'{base}.lora_B.weight'] = (64, bad_rank or r) if (layer, m) == (0, 'q_proj') else (64, r)
    if drop:
        shapes.pop(drop)
    return shapes


def test_adapter_export_key_mapping_and_pair_checks():
    export = script('export_rl_adapter')
    key = 'base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight'
    assert export.adapter_name(key) == ('base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight', 3, 'q_proj', 'A')
    with pytest.raises(ValueError, match='unexpected_lora_key'):
        export.adapter_name('base_model.model.lm_head.lora_A.default.weight')
    assert export.checked_pairs(lora_shapes(), 32, 2) == sorted(MODULES)
    with pytest.raises(ValueError, match='incomplete_lora_pair'):
        export.checked_pairs(lora_shapes(drop='base_model.model.model.layers.1.mlp.up_proj.lora_B.weight'), 32, 2)
    with pytest.raises(ValueError, match='lora_rank_mismatch'):
        export.checked_pairs(lora_shapes(bad_rank=16), 32, 2)
    with pytest.raises(ValueError, match='not_uniform_over_all_layers'):
        export.checked_pairs(lora_shapes(), 32, 3)


def test_generate_accepts_only_intact_rl_adapter_exports(tmp_path):
    from posttrain.common import write_json
    from posttrain.training.generate import checked_rl_adapter
    cfg = json.loads((ROOT/'configs/generation_pilot.json').read_text(encoding='utf-8'))
    folder = tmp_path/'export'; (folder/'adapter').mkdir(parents=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        (folder/'adapter'/name).write_text(name, encoding='utf-8')
    record = {'scope': 'rl_lora_adapter_export', 'r': 32,
              'base': {'model_id': cfg['model_id'], 'model_revision': cfg['model_revision']},
              'files': {f'adapter/{n}': file_digest(folder/'adapter'/n)
                        for n in ('adapter_config.json', 'adapter_model.safetensors')}}
    write_json(folder/'export.json', record)
    assert checked_rl_adapter(folder, cfg)['export_sha256'] == file_digest(folder/'export.json')
    write_json(folder/'export.json', {**record, 'r': 64})
    with pytest.raises(ValueError, match='max_lora_rank'):
        checked_rl_adapter(folder, cfg)
    write_json(folder/'export.json', record)
    (folder/'adapter/adapter_model.safetensors').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='changed_since_export'):
        checked_rl_adapter(folder, cfg)


def test_lineage_tokens_count_latest_settled_rollout_per_step():
    from posttrain.training.verl_pilot_hooks import lineage_generated_tokens
    def row(session, step, actual):
        return {'actual': actual, 'payload': {'purpose': f'pilot:{session}:{step}'}}
    # session a ran steps 1-3 (step 3 crashed before settling); resume b re-ran step 3 and 4
    snap = {'reservations': [row('a', 1, 10), row('a', 2, 20), row('a', 3, None), row('b', 3, 30), row('b', 4, 40),
                             {'actual': 99, 'payload': {'purpose': 'replacement:x'}}]}
    assert lineage_generated_tokens(snap, 4) == 60
    assert lineage_generated_tokens(snap, 5) == 100
    snap['reservations'].insert(3, row('a', 3, 1000))  # an older settled attempt is superseded by the resume
    assert lineage_generated_tokens(snap, 4) == 60


def _target_trainer(tmp_path, used, saved_step=None):
    from types import SimpleNamespace
    from posttrain.training.verl_pilot_hooks import instrumented_trainer
    calls = []
    snap = {'remaining': 5, 'reservations': [{'actual': used, 'payload': {'purpose': 'pilot:s:1'}}]}
    trace = SimpleNamespace(root=tmp_path, session='s', ledger=SimpleNamespace(snapshot=lambda: snap),
                            snapshot=lambda: {}, phase=lambda *a, **k: None)
    ckpt = tmp_path/'ckpt'; ckpt.mkdir()
    if saved_step is not None:
        (ckpt/'latest_checkpointed_iteration.txt').write_text(str(saved_step))
    class Base:
        def __init__(self):
            self.global_steps, self._last_synced_step = 2, 1
            self._last_complete_data_state = {'batch': 1}
            self.train_dataloader = SimpleNamespace(load_state_dict=lambda s: calls.append(('restore', s)))
            self.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(ckpt)))
            self.async_rollout_manager = SimpleNamespace(generate_sequences=lambda b: calls.append('generated'))
            self.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None)
        def init_workers(self): pass
        def fit(self): self.async_rollout_manager.generate_sequences([0, 0])
    trainer = instrumented_trainer(Base)(pilot_trace=trace, pilot_settings={'scope': 'engineering_pilot_only', 'g': 2,
        'max_response_tokens': 4, 'stop_generated_tokens': 100})
    trainer.init_workers()
    trainer._save_checkpoint = lambda: calls.append(f'save:{trainer.global_steps}')
    return trainer, calls


@pytest.mark.parametrize('saved_step', [None, 1])
def test_generated_token_target_stops_before_next_rollout_and_saves_once(tmp_path, saved_step):
    trainer, calls = _target_trainer(tmp_path, used=100, saved_step=saved_step)
    trainer.fit()
    assert 'generated' not in calls and ('restore', {'batch': 1}) in calls
    assert calls.count('save:1') == (1 if saved_step is None else 0)
    done = json.loads((tmp_path/'completion.json').read_text())
    assert done['completion_reason'] == 'generated_token_target_reached' and done['last_complete_update'] == 1
    assert done['lineage_generated_tokens'] == 100 and trainer.global_steps == 2


def test_generated_token_target_not_reached_keeps_training(tmp_path):
    trainer, calls = _target_trainer(tmp_path, used=99)
    # Below the target the hook proceeds to the normal reservation (absent from this minimal trace fixture).
    with pytest.raises(AttributeError, match='begin_rollout'):
        trainer.fit()
    assert not (tmp_path/'completion.json').exists()


def test_stop_generated_tokens_setting_is_validated():
    from posttrain.training.code_rl import checked_settings
    cfg = json.loads((ROOT/'configs/single_rl_kodcode3m_v4.json').read_text(encoding='utf-8'))
    checked_settings({**cfg, 'stop_generated_tokens': 1000000})
    for bad in (0, 1.5, cfg['token_cap'] + 1):
        with pytest.raises(ValueError, match='invalid_stop_generated_tokens'):
            checked_settings({**cfg, 'stop_generated_tokens': bad})


def test_lr_warmup_setting_is_validated():
    from posttrain.training.code_rl import checked_settings
    cfg = json.loads((ROOT/'configs/single_rl_kodcode3m_v4.json').read_text(encoding='utf-8'))
    checked_settings({**cfg, 'lr_warmup_steps': 10}); checked_settings({**cfg, 'lr_warmup_steps': 0})
    for bad in (-1, 2.5, '10'):
        with pytest.raises(ValueError, match='invalid_lr_warmup_steps'):
            checked_settings({**cfg, 'lr_warmup_steps': bad})
