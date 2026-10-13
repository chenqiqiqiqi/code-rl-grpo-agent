"""Pinned-source, CPU-only regression for the observed NO_SHARD sync failure."""
import ast
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from posttrain.training import code_rl, verl_pilot

SOURCE = Path('.cache/upstream/verl-source') / ('verl-' + verl_pilot.VERL_REVISION)


@pytest.mark.parametrize('mode', ['single', 'agent'])
def test_single_gpu_sync_avoids_unsupported_offload_and_keeps_all_adapters(tmp_path, monkeypatch, mode):
    if not SOURCE.is_dir():
        pytest.skip('requires the pinned local verl source snapshot')
    monkeypatch.syspath_prepend(str(SOURCE.resolve()))
    settings = json.loads(Path(f'configs/{mode}_rl_pilot.json').read_text())
    config = code_rl.compose(settings, SOURCE, tmp_path/'model', tmp_path/'data',
                             tmp_path, tmp_path/'context.json', 'a'*64)
    # Run the actual pinned collector with a NO_SHARD contract double. No tensors,
    # candidate programs, CUDA workers or upstream module side effects are run.
    tree = ast.parse((SOURCE/'verl/utils/fsdp_utils.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_lora_params')
    calls = []

    @contextmanager
    def summon(module, writeback=False, offload_to_cpu=False):
        calls.append(offload_to_cpu)
        if offload_to_cpu:
            raise NotImplementedError('offload_to_cpu=True and NO_SHARD is not supported yet')
        yield

    class Tensor:
        def detach(self): return self
        def cpu(self): return self

    adapters = {'layer.q_proj.lora_A.weight': Tensor(), 'layer.q_proj.lora_B.weight': Tensor()}
    peft = ModuleType('peft.utils.save_and_load')
    peft.get_peft_model_state_dict = lambda model: dict(adapters)
    monkeypatch.setitem(sys.modules, 'peft.utils.save_and_load', peft)
    namespace = dict(OrderedDict=OrderedDict, FSDP=SimpleNamespace(summon_full_params=summon),
        fsdp_version=lambda module: 1, layered_summon_lora_params=lambda module: {},
        get_torch_device=lambda: SimpleNamespace(empty_cache=lambda: None), nullcontext=nullcontext)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE/'verl/utils/fsdp_utils.py'), 'exec'), namespace)
    collect = namespace['collect_lora_params']
    with pytest.raises(NotImplementedError, match='NO_SHARD'):
        collect(object(), layered_summon=True, base_sync_done=True)
    calls.clear()
    result = collect(object(), layered_summon=config.actor_rollout_ref.rollout.layered_summon,
                     base_sync_done=True)
    assert result == adapters
    assert calls == [False]
    assert config.actor_rollout_ref.actor.strategy == 'fsdp'
    assert config.actor_rollout_ref.rollout.load_format == 'safetensors'
    assert config.trainer.n_gpus_per_node == 1
