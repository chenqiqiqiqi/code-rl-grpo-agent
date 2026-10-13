#!/usr/bin/env bash
# Usage (GPU host, setsid -f): final_ktest.sh
# One-time final evaluation on the KodCode test set (protocol v4 addenda 9/10; user 2026-10-02: KodCode only):
# single-turn k=8 seed 42 (same sampling as development) for M0, W (kodcode2m@124), C (phase2c@160), A (phase2a@34).
# Outputs runs/stage4-phase1/final-ktest-NAME(-cands.jsonl); runs under the GPU lock; skips finished outputs.
set -uo pipefail
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH=$PWD/src:$V VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; P=$V/.venv/bin/python
for spec in m0: w:kodcode2m-export-124 c:phase2c-export-160 a:phase2a-export-34; do
  NAME=${spec%%:*}; AD=${spec#*:}; OUT=$R/final-ktest-$NAME
  [ -f $OUT/report.json ] || flock runs/claude-diag-20260930/gpu.lock $P -m posttrain.training.generate \
    --config stage4-phase1/gen_kodcode_test_k8_s42.json --prompts data/kodcode-rl-v1-public/gpu-test-prompts.jsonl \
    --output $OUT --model-snapshot $M ${AD:+--rl-adapter $R/$AD} > $OUT.log 2>&1 || { echo "$(date -Is) final-ktest-$NAME FAILED" >> $R/sweep.done; continue; }
  $P claude-diag-20260930/strip_cands.py $OUT/candidates.jsonl $OUT-cands.jsonl
  echo "$(date -Is) final-ktest-$NAME ready" >> $R/sweep.done
done
