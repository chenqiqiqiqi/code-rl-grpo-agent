#!/usr/bin/env bash
# Usage (GPU host): run_verl_formal.sh NAME CONFIG FREEZE EVIDENCE_DIR [RESUME_RECEIPT]
# Stage 4 phase 1 formal single-turn GRPO (protocol v4, M1 = M0) on the 4090D. Serialised by the
# shared GPU lock; samples GPU memory every second. Start with setsid nohup; resume with a receipt
# from runs/stage4-phase1/NAME/audit/checkpoints/.
set -uo pipefail
NAME=$1; CONFIG=$2; FREEZE=$3; E=$4; RESUME=${5:-}
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
R=runs/stage4-phase1; OUT=$R/$NAME; mkdir -p $R
EXTRA=""; [ -n "$RESUME" ] && EXTRA="--resume-receipt $RESUME"
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) &
MON=$!
echo "$(date -Is) start config=$CONFIG freeze=$FREEZE evidence=$E resume=${RESUME:-none}" >> $R/$NAME.exit
flock runs/claude-diag-20260930/gpu.lock "$V/.venv/bin/python" -m posttrain.training.code_rl --verl-source "$V" \
  --model-snapshot /root/autodl-tmp/models/qwen7b --prepared data/prepared/rl-pool-v5b \
  --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
  --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json \
  --config $CONFIG --freeze $FREEZE --output $OUT $EXTRA >> $R/$NAME-run.log 2>&1
RC=$?
kill $MON 2>/dev/null
echo "$(date -Is) exit $RC" >> $R/$NAME.exit
echo "exit $RC; peak MiB $(cut -d, -f2 $R/$NAME-gpu-memory.csv | tr -dc '0-9\n' | sort -n | tail -1)"
