#!/usr/bin/env bash
# Usage (GPU host, setsid -f): phase2_screen_single.sh RUN STEP NAME   (e.g. kodcode2m 124 w124-shard2-single)
# Phase-2 single-turn screening (protocol v4 addenda 6/7): W = RUN@STEP adapter, all 1935 shard-2 prompts, k=4,
# T=0.7, seed 42 -> runs/stage4-phase1/NAME(-cands.jsonl). Runs under the GPU lock.
set -uo pipefail
RUN=$1; S=$2; NAME=$3
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH=$PWD/src:$V VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; P=$V/.venv/bin/python; OUT=$R/$NAME
[ -f $OUT/report.json ] || flock runs/claude-diag-20260930/gpu.lock $P -m posttrain.training.generate \
  --config stage4-phase1/gen_kodcode_shard2_k4.json --prompts data/kodcode-rl-v2-public/gpu-phase2-prompts.jsonl \
  --output $OUT --model-snapshot $M --rl-adapter $R/$RUN-export-$S > $OUT.log 2>&1 || { echo "$(date -Is) $NAME FAILED" >> $R/sweep.done; exit 1; }
$P claude-diag-20260930/strip_cands.py $OUT/candidates.jsonl $OUT-cands.jsonl
echo "$(date -Is) $NAME ready ($RUN@$S)" >> $R/sweep.done
