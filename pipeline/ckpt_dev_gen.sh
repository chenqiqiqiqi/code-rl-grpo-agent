#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): ckpt_dev_gen.sh RUN STEP [RUN STEP ...]
# Learning-curve diagnostic: export the adapter of an existing checkpoint and generate development k=8
# seed-42 candidates with it (same config as the M0 baseline); writes RUN@STEP-dev-s42-cands.jsonl.
set -uo pipefail
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$V/.venv/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b
while [ $# -ge 2 ]; do
  RUN=$1; STEP=$2; shift 2
  [ -d $R/$RUN-export-$STEP ] || python scripts/export_rl_adapter.py --run-dir $R/$RUN --model-snapshot $M \
    --output $R/$RUN-export-$STEP --receipt "$(ls -t $R/$RUN/audit/checkpoints/*-$STEP.json | head -1)" || continue
  [ -f $R/$RUN-at$STEP-dev-s42/report.json ] || flock runs/claude-diag-20260930/gpu.lock python -m posttrain.training.generate \
    --config stage4-phase1/gen_dev_k8_s42.json --prompts data/prepared/rl-pool-v5b/single-development.jsonl \
    --output $R/$RUN-at$STEP-dev-s42 --model-snapshot $M --rl-adapter $R/$RUN-export-$STEP || continue
  python claude-diag-20260930/strip_cands.py $R/$RUN-at$STEP-dev-s42/candidates.jsonl $R/$RUN-at$STEP-dev-s42-cands.jsonl
  echo "$(date -Is) $RUN@$STEP ready" >> $R/ckpt-dev.done
done
