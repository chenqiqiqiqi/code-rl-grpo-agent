"""Fixed-token check of an exported RL adapter against the training run's own records.

Checkpoint k holds the weights after update k; the rollout and the actor's recomputed
old_log_prob of step k+1 used exactly those weights. Offline vLLM + exported adapter must
reproduce them on the same tokens; the base model without the adapter is the contrast.
GPU-host side; run only when no training job holds the GPU.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from posttrain.common import file_digest,utc_now,write_json


def step_rows(run_dir,step):
    """Join the step's rollout tokens with the actor's old_log_prob, one session only."""
    olds=sorted((run_dir/'audit/old_log_probs').glob(f'*-{step}.json'))
    if not olds:raise ValueError('no_old_log_probs_for_step')
    old=json.loads(olds[0].read_text(encoding='utf-8'))
    rollouts=[json.loads(p.read_text(encoding='utf-8')) for p in (run_dir/'audit/rollouts').glob('*.json')]
    rollout=[r for r in rollouts if r['session']==old['session'] and r['step']==step]
    if len(rollout)!=1:raise ValueError('ambiguous_or_missing_rollout_for_step')
    candidates={c['request_id']:c for c in rollout[0]['candidates']}
    rows=[]
    for row in old['rows']:
        c=candidates[row['request_id']]
        if c['response_tokens']!=row['tokens']:raise ValueError('rollout_and_old_log_prob_tokens_differ')
        rows.append({'request_id':row['request_id'],'prompt':c['prompt_tokens'],'response':row['tokens'],
                     'old':row['old_log_probs'],'rollout':c['rollout_log_probs']})
    return rows,{'old_log_probs':str(olds[0]),'session':old['session']}


def response_log_probs(llm,rows,lora=None):
    from vllm import SamplingParams
    # prompt_logprobs=1 returns top-1 plus the actual prompt token at every position.
    params=SamplingParams(max_tokens=1,temperature=1.0,prompt_logprobs=1)
    outputs=llm.generate([{'prompt_token_ids':r['prompt']+r['response']} for r in rows],params,
                         lora_request=lora,use_tqdm=False)
    result=[]
    for row,out in zip(rows,outputs,strict=True):
        tokens=row['prompt']+row['response'];start=len(row['prompt'])
        result.append([out.prompt_logprobs[i][tokens[i]].logprob for i in range(start,len(tokens))])
    return result


def mean_abs(a,b):
    diffs=[abs(x-y) for xs,ys in zip(a,b,strict=True) for x,y in zip(xs,ys,strict=True)]
    return {'mean':statistics.fmean(diffs),'max':max(diffs),'tokens':len(diffs)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--export',type=Path,required=True,help='export_rl_adapter.py output (checkpoint k)')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=0.01)
    p.add_argument('--step',type=int,help='compare this step of --run-dir (default: export step + 1, same run)')
    a=p.parse_args()
    record=json.loads((a.export/'export.json').read_text(encoding='utf-8'))
    for name,sha in record['files'].items():
        if file_digest(a.export/name)!=sha:raise ValueError('adapter_changed_since_export')
    step=a.step if a.step is not None else record['global_step']+1
    rows,source=step_rows(a.run_dir,step)
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    longest=max(len(r['prompt'])+len(r['response']) for r in rows)
    llm=LLM(model=str(a.model_snapshot),tokenizer=str(a.model_snapshot),dtype='bfloat16',enable_lora=True,
            max_lora_rank=record['r'],enforce_eager=True,max_model_len=longest+1,gpu_memory_utilization=0.85,seed=0)
    adapter=response_log_probs(llm,rows,LoRARequest('rl_export',1,str((a.export/'adapter').resolve())))
    base=response_log_probs(llm,rows)
    old=[r['old'] for r in rows];rollout=[r['rollout'] for r in rows]
    report={'created_at':utc_now(),'export_sha256':file_digest(a.export/'export.json'),'checkpoint_step':record['global_step'],
        'compared_step':step,'source':source,'sequences':len(rows),
        'adapter_vs_actor_old':mean_abs(adapter,old),'adapter_vs_training_rollout':mean_abs(adapter,rollout),
        'base_vs_actor_old':mean_abs(base,old),'training_rollout_vs_actor_old':mean_abs(rollout,old),
        'threshold_mean_abs':a.threshold}
    report['passed']=report['adapter_vs_actor_old']['mean']<=a.threshold
    report['adapter_effect_detectable']=report['base_vs_actor_old']['mean']>2*report['adapter_vs_actor_old']['mean']
    write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('passed','adapter_effect_detectable','adapter_vs_actor_old','base_vs_actor_old')}))


if __name__=='__main__':main()
