"""Export the LoRA adapter of a verl FSDP actor checkpoint for offline vLLM evaluation.

GPU-host side (needs torch/safetensors/peft from the pinned verl venv). The checkpoint
is authenticated by its receipt first; only lora_A/lora_B tensors are exported, with
rank/alpha taken from the run's resolved config (never defaulted). The base weights
stay the hash-bound official snapshot, so no 15 GB merged copy is written.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from posttrain.common import file_digest,utc_now,write_json
from posttrain.training.verl_pilot import MODEL,MODEL_REVISION
from posttrain.training.verl_pilot_trace import verify_checkpoint

KEY=re.compile(r'base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight')


def adapter_name(key):
    """verl/PEFT FSDP state keys carry the adapter name; PEFT files do not."""
    name=key.replace('.default.weight','.weight')
    match=KEY.fullmatch(name)
    if not match:raise ValueError('unexpected_lora_key:'+key)
    return name,int(match.group(1)),match.group(3),match.group(4)


def checked_pairs(shapes,rank,layers):
    """shapes: {peft_key: (rows, cols)}. Every module needs both A (r x in) and B (out x r)."""
    modules={}
    for key,shape in shapes.items():
        _,layer,module,side=adapter_name(key)
        if len(shape)!=2:raise ValueError('lora_tensor_not_matrix:'+key)
        if (side=='A' and shape[0]!=rank) or (side=='B' and shape[1]!=rank):raise ValueError('lora_rank_mismatch:'+key)
        modules.setdefault((layer,module),set()).add(side)
    if any(sides!={'A','B'} for sides in modules.values()):raise ValueError('incomplete_lora_pair')
    per_layer={}
    for layer,module in modules:per_layer.setdefault(layer,set()).add(module)
    if set(per_layer)!=set(range(layers)) or len({frozenset(m) for m in per_layer.values()})!=1:
        raise ValueError('lora_modules_not_uniform_over_all_layers')
    return sorted(next(iter(per_layer.values())))


def local_tensor(value):
    if hasattr(value,'_local_tensor'):tensor=value._local_tensor  # DTensor, world size 1
    elif hasattr(value,'local_shards'):
        shards=value.local_shards()
        if len(shards)!=1:raise ValueError('single_rank_checkpoint_required')
        tensor=shards[0].tensor
    else:tensor=value
    if tuple(tensor.shape)!=tuple(value.shape):raise ValueError('sharded_lora_tensor_not_complete')
    return tensor.detach().contiguous()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True,help='code_rl --output directory')
    p.add_argument('--receipt',type=Path,required=True,help='audit/checkpoints/<session>-<step>.json')
    p.add_argument('--model-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('immutable_export_output_exists')
    receipt=json.loads(a.receipt.read_text(encoding='utf-8'))
    checkpoint=Path(receipt['checkpoint'])
    verify_checkpoint(checkpoint,receipt['receipt'])
    resolved=json.loads((a.run_dir/'resolved-config.json').read_text(encoding='utf-8'))
    model_cfg=resolved['actor_rollout_ref']['model']
    rank,alpha=model_cfg['lora_rank'],model_cfg['lora_alpha']
    if type(rank) is not int or type(alpha) is not int or rank<1 or alpha<1:raise ValueError('run_lora_config_required')
    meta_path=checkpoint/'actor/lora_train_meta.json'
    if meta_path.is_file():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('r')!=rank or meta.get('lora_alpha')!=alpha:raise ValueError('checkpoint_lora_meta_mismatch')
    layers=json.loads((a.model_snapshot/'config.json').read_text(encoding='utf-8'))['num_hidden_layers']
    import torch
    from safetensors.torch import save_file
    import peft
    state=torch.load(checkpoint/'actor/model_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
    tensors={}
    for key,value in state.items():
        if 'lora_' not in key:continue
        name=adapter_name(key)[0]
        if name in tensors:raise ValueError('duplicate_lora_key:'+name)
        tensors[name]=local_tensor(value)
    if not tensors:raise ValueError('no_lora_tensors_in_checkpoint')
    if any(not torch.isfinite(t.float()).all() for t in tensors.values()):raise ValueError('nonfinite_lora_tensor')
    modules=checked_pairs({k:tuple(v.shape) for k,v in tensors.items()},rank,layers)
    config=peft.LoraConfig(r=rank,lora_alpha=alpha,target_modules=modules,task_type='CAUSAL_LM',
                           lora_dropout=0.0,bias='none').to_dict()
    for key in ('task_type','peft_type'):
        config[key]=getattr(config[key],'value',config[key])
    config['target_modules']=sorted(config['target_modules'])
    config['base_model_name_or_path']=MODEL;config['revision']=MODEL_REVISION
    folder=a.output/'adapter';folder.mkdir(parents=True)
    (folder/'adapter_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    save_file(tensors,str(folder/'adapter_model.safetensors'))
    step=int(checkpoint.name.rsplit('_',1)[1])
    write_json(a.output/'export.json',{'scope':'rl_lora_adapter_export','created_at':utc_now(),
        'base':{'model_id':MODEL,'model_revision':MODEL_REVISION},
        'run_identity':json.loads((a.run_dir/'audit/identity.json').read_text(encoding='utf-8')),
        'checkpoint':str(checkpoint.resolve()),'global_step':step,
        'receipt_path':str(a.receipt.resolve()),'receipt_sha256':file_digest(a.receipt),
        'r':rank,'lora_alpha':alpha,'target_modules':modules,'tensors':len(tensors),
        'lora_dtype':str(next(iter(tensors.values())).dtype),
        'files':{f'adapter/{n}':file_digest(folder/n) for n in ('adapter_config.json','adapter_model.safetensors')}})
    print(json.dumps({'global_step':step,'tensors':len(tensors),'modules':modules,'r':rank,'alpha':alpha}))


if __name__=='__main__':main()
