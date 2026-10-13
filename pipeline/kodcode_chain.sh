#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): kodcode_chain.sh EVIDENCE_DIR NAME
# KodCode RL experiment: train configs/single_rl_NAME_v4.json on data/prepared/kodcode-rl-v1 from M0
# (engineering scope, fixed steps), export the final adapter, then generate k=8 seed-42 candidates on
# the 155 held-out KodCode problems and on the TACO development set (diagnostic).
set -uo pipefail
E=$1; NAME=$2
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock; OUT=$R/$NAME
STEPS=$(python -c "import json;print(json.load(open('configs/single_rl_${NAME}_v4.json'))['total_steps'])")
if [ "$(cat $OUT/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null)" != "$STEPS" ]; then
  ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) & MON=$!
  echo "$(date -Is) start $NAME" >> $R/$NAME.exit
  flock $LOCK python -m posttrain.training.code_rl --verl-source "$V" --model-snapshot $M --prepared data/prepared/kodcode-rl-v1 \
    --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
    --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json \
    --config configs/single_rl_${NAME}_v4.json --output $OUT >> $R/$NAME-run.log 2>&1
  RC=$?; kill $MON 2>/dev/null; echo "$(date -Is) exit $RC" >> $R/$NAME.exit
  [ $RC -eq 0 ] || exit $RC
fi
python scripts/export_rl_adapter.py --run-dir $OUT --model-snapshot $M --output $R/$NAME-export-$STEPS \
  --receipt "$(ls -t $OUT/audit/checkpoints/*-$STEPS.json | head -1)" || [ -d $R/$NAME-export-$STEPS ] || exit 1
for EV in kodcode-probe:stage4-phase1/gen_kodcode_probe_k8.json:data/kodcode-rl-v1-public/gpu-probe-prompts.jsonl \
          dev-s42:stage4-phase1/gen_dev_k8_s42.json:data/prepared/rl-pool-v5b/single-development.jsonl; do
  IFS=: read TAG CFG PROMPTS <<< "$EV"
  [ -f $R/$NAME-$TAG/report.json ] || flock $LOCK python -m posttrain.training.generate --config $CFG --prompts $PROMPTS \
    --output $R/$NAME-$TAG --model-snapshot $M --rl-adapter $R/$NAME-export-$STEPS >> $R/$NAME-eval-gen.log 2>&1 || exit 1
  python claude-diag-20260930/strip_cands.py $R/$NAME-$TAG/candidates.jsonl $R/$NAME-$TAG-cands.jsonl
done
echo "$(date -Is) $NAME eval candidates ready" >> $R/$NAME.done
