#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): ext_resume.sh EVIDENCE_DIR NAME [NAME ...]
# Resume each phase-1 extension run (configs/single_rl_NAME_v4.json, data/prepared/kodcode-rl-v2) from its
# latest checkpoint receipt until step 60, retrying up to 3 times (e.g. sporadic OOM near the 48 GB limit),
# then export checkpoints 20/40/60 and generate KodCode development candidates (k=8, seed 42).
set -uo pipefail
E=$1; shift
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
clean_gpu(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill $p 2>/dev/null; done; sleep 5; }
for NAME in "$@"; do
  OUT=$R/$NAME
  for attempt in 1 2 3; do
    [ "$(cat $OUT/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null)" = "60" ] && break
    clean_gpu
    REC=$(ls -t $OUT/audit/checkpoints/*.json | head -1)
    ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) & MON=$!
    echo "$(date -Is) resume $NAME from $REC (attempt $attempt)" >> $R/$NAME.exit
    flock $LOCK python -m posttrain.training.code_rl --verl-source "$V" --model-snapshot $M --prepared data/prepared/kodcode-rl-v2 \
      --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
      --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json \
      --config configs/single_rl_${NAME}_v4.json --output $OUT --resume-receipt $REC >> $R/$NAME-run.log 2>&1
    RC=$?; kill $MON 2>/dev/null; echo "$(date -Is) resume-exit $RC" >> $R/$NAME.exit
  done
  clean_gpu
  bash stage4-phase1/ckpt_gen.sh 999999 s42 $NAME 20 $NAME 40 $NAME 60
done
echo "$(date -Is) ext resume done: $*" >> $R/sweep.done
