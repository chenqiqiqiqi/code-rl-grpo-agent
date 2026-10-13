#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): probe_chain.sh EVIDENCE_DIR NAME...   e.g. probe_chain.sh evidence/stage4-s4b probe1 probe2
# Low-cost probes (STAGE4_PHASE1_PLAN_DRAFT.md): each NAME runs configs/single_rl_NAME_v4.json from M0 for its fixed
# steps (engineering scope, no freeze), then exports the final adapter and generates k=8 seed-42 candidates on the
# held-out training probe set. Probes run strictly one after another under the shared GPU lock.
set -uo pipefail
E=$1; shift
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH="$PWD/src:$V" TOKENIZERS_PARALLELISM=false CUDA_HOME=/usr/local/cuda-13.0 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$V/.venv/bin:$CUDA_HOME/bin:$PATH"
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
for NAME in "$@"; do
  OUT=$R/$NAME
  if [ ! -f $OUT/checkpoints/latest_checkpointed_iteration.txt ] || [ "$(cat $OUT/checkpoints/latest_checkpointed_iteration.txt)" != "$(python -c "import json;print(json.load(open('configs/single_rl_${NAME}_v4.json'))['total_steps'])")" ]; then
    ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $R/$NAME-gpu-memory.csv; sleep 1; done ) & MON=$!
    echo "$(date -Is) start $NAME" >> $R/$NAME.exit
    flock $LOCK python -m posttrain.training.code_rl --verl-source "$V" --model-snapshot $M --prepared data/prepared/rl-pool-v5b \
      --model-files manifests/qwen7b_model_files.json --gpu-probe runs/verl-4090d-20260930/environment.json \
      --attestation $E/isolation/attestation.json --gateway-identity $E/gateway_identity.json \
      --config configs/single_rl_${NAME}_v4.json --output $OUT >> $R/$NAME-run.log 2>&1
    RC=$?; kill $MON 2>/dev/null; echo "$(date -Is) exit $RC" >> $R/$NAME.exit
    [ $RC -eq 0 ] || { echo "$NAME failed rc=$RC"; exit $RC; }
  fi
  STEP=$(cat $OUT/checkpoints/latest_checkpointed_iteration.txt)
  [ -d $R/$NAME-export-$STEP ] || python scripts/export_rl_adapter.py --run-dir $OUT --model-snapshot $M \
    --output $R/$NAME-export-$STEP --receipt "$(ls -t $OUT/audit/checkpoints/*-$STEP.json | head -1)" || exit 1
  [ -f $R/$NAME-probe-s42/report.json ] || flock $LOCK python -m posttrain.training.generate --config stage4-phase1/gen_probe_k8_s42.json \
    --prompts data/prepared/rl-pool-v5b-probe/probe-train.jsonl --output $R/$NAME-probe-s42 --model-snapshot $M \
    --rl-adapter $R/$NAME-export-$STEP >> $R/$NAME-probe-gen.log 2>&1 || exit 1
  python claude-diag-20260930/strip_cands.py $R/$NAME-probe-s42/candidates.jsonl $R/$NAME-probe-s42-cands.jsonl
  echo "$(date -Is) $NAME probe candidates ready" >> $R/probe-chain.done
done
