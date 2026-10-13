#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): ckpt_gen.sh WAIT_PID SEED_CONFIG_TAG RUN STEP [RUN STEP ...]
# After WAIT_PID exits: for each RUN STEP, export the checkpoint adapter (if needed) and generate k=8
# candidates on the KodCode development set with the given generation config tag:
#   s42 -> stage4-phase1/gen_kodcode_dev_k8_s42.json, output RUN-atSTEP-kdev
#   s43 -> stage4-phase1/gen_kodcode_dev_k8_s43.json, output RUN-atSTEP-kdev-s43
set -uo pipefail
WAIT=$1; TAG=$2; shift 2
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH=$PWD/src:$V VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; P=$V/.venv/bin/python
while kill -0 "$WAIT" 2>/dev/null; do sleep 15; done
SUF=$([ "$TAG" = s42 ] && echo kdev || echo kdev-$TAG)
while [ $# -ge 2 ]; do
  RUN=$1; S=$2; shift 2
  [ -d $R/$RUN-export-$S ] || $P scripts/export_rl_adapter.py --run-dir $R/$RUN --model-snapshot $M \
    --output $R/$RUN-export-$S --receipt "$(ls -t $R/$RUN/audit/checkpoints/*-$S.json | head -1)" || continue
  [ -f $R/$RUN-at$S-$SUF/report.json ] || flock runs/claude-diag-20260930/gpu.lock $P -m posttrain.training.generate \
    --config stage4-phase1/gen_kodcode_dev_k8_$TAG.json --prompts data/kodcode-rl-v1-public/gpu-dev-prompts.jsonl \
    --output $R/$RUN-at$S-$SUF --model-snapshot $M --rl-adapter $R/$RUN-export-$S || continue
  $P claude-diag-20260930/strip_cands.py $R/$RUN-at$S-$SUF/candidates.jsonl $R/$RUN-at$S-$SUF-cands.jsonl
  echo "$(date -Is) $RUN@$S $SUF ready" >> $R/sweep.done
done
