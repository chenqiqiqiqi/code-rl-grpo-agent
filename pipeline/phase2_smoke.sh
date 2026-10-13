#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): phase2_smoke.sh NAME CONFIG PREPARED EVIDENCE_DIR
# Phase-2 smoke run (token-target stop / Agent memory): one code_rl run under the GPU lock with a 1 s
# memory log. Never kills other GPU processes (other jobs wait on the same lock).
set -uo pipefail
NAME=$1; CFG=$2; PREP=$3; E=$4
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
TOOL=(); grep -q '"mode": "agent"' $CFG && TOOL=(--tool-identity $E/tool_identity.json)
flock $LOCK bash -c "( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) & MON=\$!
  echo \"\$(date -Is) start $NAME\" >> $R/$NAME.exit
  python -m posttrain.training.code_rl --verl-source $V --model-snapshot $M --prepared $PREP \
    --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
    --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json ${TOOL[*]} \
    --config $CFG --output $R/$NAME >> $R/$NAME-run.log 2>&1
  RC=\$?; kill \$MON; echo \"\$(date -Is) exit \$RC\" >> $R/$NAME.exit"
