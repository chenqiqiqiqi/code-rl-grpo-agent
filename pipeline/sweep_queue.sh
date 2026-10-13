#!/usr/bin/env bash
# Usage (GPU host, setsid nohup): sweep_queue.sh WAIT_PID
# Step-sweep plan (user, 2026-10-01 evening): after WAIT_PID (M0 KodCode-dev generation) ends:
#   KodCode binary sweep (124 steps, all checkpoints) -> its checkpoint candidates on the KodCode dev set
#   KodCode mixed sweep  (124 steps, all checkpoints) -> its checkpoint candidates on the KodCode dev set
#   TACO binary sweep (5e-5, 4x8, 160 steps, all checkpoints) -> its checkpoint candidates on the TACO dev set
# The VM scores candidates separately (stage4-phase1/vm_job_runner.sh), so the GPU never waits for it.
cd /root/autodl-tmp/posttrain-project
V=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b
export PYTHONPATH=$PWD/src:$V VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
R=runs/stage4-phase1; M=/root/autodl-tmp/models/qwen7b; LOCK=runs/claude-diag-20260930/gpu.lock
while kill -0 "$1" 2>/dev/null; do sleep 10; done
$V/.venv/bin/python claude-diag-20260930/strip_cands.py $R/m0-kodcode-dev/candidates.jsonl $R/m0-kodcode-dev-cands.jsonl && touch $R/m0-kodcode-dev.ready

ckpt_gen(){  # RUN CONFIG PROMPTS TAG STEP...
  local RUN=$1 CFG=$2 PROMPTS=$3 TAG=$4; shift 4
  for S in "$@"; do
    [ -d $R/$RUN-export-$S ] || $V/.venv/bin/python scripts/export_rl_adapter.py --run-dir $R/$RUN --model-snapshot $M \
      --output $R/$RUN-export-$S --receipt "$(ls -t $R/$RUN/audit/checkpoints/*-$S.json | head -1)" || continue
    [ -f $R/$RUN-at$S-$TAG/report.json ] || flock $LOCK $V/.venv/bin/python -m posttrain.training.generate --config $CFG \
      --prompts $PROMPTS --output $R/$RUN-at$S-$TAG --model-snapshot $M --rl-adapter $R/$RUN-export-$S || continue
    $V/.venv/bin/python claude-diag-20260930/strip_cands.py $R/$RUN-at$S-$TAG/candidates.jsonl $R/$RUN-at$S-$TAG-cands.jsonl
    echo "$(date -Is) $RUN@$S $TAG ready" >> $R/sweep.done
  done
}
KD=data/kodcode-rl-v1-public/gpu-dev-prompts.jsonl; KC=stage4-phase1/gen_kodcode_dev_k8_s42.json
TD=data/prepared/rl-pool-v5b/single-development.jsonl; TC=stage4-phase1/gen_dev_k8_s42.json
for RUN in kodcode2b kodcode2m; do
  bash stage4-phase1/kodcode_chain.sh evidence/stage4-s4c $RUN
  ckpt_gen $RUN $KC $KD kdev 20 40 60 80 100 124
done
bash stage4-phase1/probe_chain.sh evidence/stage4-s4c tacosweep
ckpt_gen tacosweep $TC $TD dev 20 40 60 80 100 120 140 160
echo "$(date -Is) sweep queue done" >> $R/sweep.done
