#!/usr/bin/env bash
# Usage (GPU host, setsid -f): phase2_train.sh EVIDENCE_DIR NAME CONFIG PREPARED [NAME CONFIG PREPARED ...]
# Phase-2 C/A training (protocol v4 addendum 6), one run after another. Each run ends normally when its
# generated-token budget is reached (audit/completion.json). A failed attempt (e.g. sporadic OOM) resumes
# from the newest checkpoint receipt, up to 4 attempts. Leftover GPU processes of a crashed attempt are
# cleaned only while this script holds the GPU lock, so other lock-respecting jobs are never killed.
# Afterwards: export the final checkpoint and generate KodCode dev candidates (k=8, seed 42).
set -uo pipefail
E=$1; shift
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
while [ $# -ge 3 ]; do
  NAME=$1; CFG=$2; PREP=$3; shift 3; OUT=$R/$NAME
  TOOL=""; grep -q '"mode": "agent"' $CFG && TOOL="--tool-identity $E/tool_identity.json"
  for attempt in 1 2 3 4; do
    [ -f $OUT/audit/completion.json ] && break
    REC=$(ls -t $OUT/audit/checkpoints/*.json 2>/dev/null | head -1)
    if [ -z "$REC" ] && [ -e $OUT ]; then
      mkdir -p $R/failed-20261002; mv $OUT $R/failed-20261002/$NAME-$(date +%s)
    fi
    RES=""; [ -n "$REC" ] && RES="--resume-receipt $REC"
    echo "$(date -Is) start $NAME attempt $attempt ${REC:+resume $REC}" >> $R/$NAME.exit
    flock $LOCK bash -c "
      for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill \$p 2>/dev/null; done; sleep 5
      ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) & MON=\$!
      python -m posttrain.training.code_rl --verl-source $V --model-snapshot $M --prepared $PREP \
        --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
        --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json $TOOL \
        --config $CFG --output $OUT $RES >> $R/$NAME-run.log 2>&1
      RC=\$?; kill \$MON; echo \"\$(date -Is) exit \$RC\" >> $R/$NAME.exit
      for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill \$p 2>/dev/null; done"
  done
  if [ -f $OUT/audit/completion.json ]; then
    S=$(python -c "import json;print(json.load(open('$OUT/audit/completion.json'))['last_complete_update'])")
    echo "$(date -Is) $NAME complete at step $S" >> $R/sweep.done
    bash stage4-phase1/ckpt_gen.sh 999999 s42 $NAME $S
  else
    echo "$(date -Is) $NAME FAILED after 4 attempts" >> $R/sweep.done
  fi
done
echo "$(date -Is) phase2 train chain done" >> $R/sweep.done
