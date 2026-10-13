#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): post_branch.sh NAME
# After a formal branch ends: export the final adapter and the last regular save before it,
# fixed-token log-prob check of the latter against the run's own step+1 records, then the
# development-set k=8 seed-42 generation with the final adapter (same config as the M0 baseline).
set -euo pipefail
NAME=$1
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$V/.venv/bin:$PATH"
R=runs/stage4-phase1; RUN=$R/$NAME; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
FINAL=$(python -c "import json;print(json.load(open('$RUN/audit/completion.json'))['last_complete_update'])")
PREV=$(( (FINAL-1)/10*10 ))
receipt(){ ls -t $RUN/audit/checkpoints/*-$1.json | head -1; }
for S in $FINAL $PREV; do
  [ -d $R/$NAME-export-$S ] || python scripts/export_rl_adapter.py --run-dir $RUN --receipt "$(receipt $S)" \
    --model-snapshot $M --output $R/$NAME-export-$S
done
[ -f $R/$NAME-export-check.json ] || flock $LOCK python scripts/check_rl_adapter_logprobs.py \
  --export $R/$NAME-export-$PREV --run-dir $RUN --model-snapshot $M --output $R/$NAME-export-check.json
[ -f $R/$NAME-dev-s42/report.json ] || flock $LOCK python -m posttrain.training.generate \
  --config stage4-phase1/gen_dev_k8_s42.json --prompts data/prepared/rl-pool-v5b/single-development.jsonl \
  --output $R/$NAME-dev-s42 --model-snapshot $M --rl-adapter $R/$NAME-export-$FINAL
echo "final=$FINAL prev=$PREV" > $R/$NAME-post.done
