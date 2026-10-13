#!/usr/bin/env bash
# Usage (local Git Bash, background): bash stage4-phase1/agent_driver_kodcode.sh TAG K SEED [ADAPTER_DIR_ON_GPU]
# KodCode version of eval-baseline-20260930/agent_driver.sh: turn-batched Agent evaluation (protocol v5b,
# function-task system prompt) on the KodCode development set. GPU generates one turn for all live
# episodes (eval-baseline-20260930/agent_eval.py), the VM runs the tool requests (claude_agent_tools_v2.py,
# current attestation), then the VM scores final submissions with the function harness (claude_score_v2.py).
# Differences from the TACO driver: no claude-jobs queue / pkill pre-emption, KodCode prompts and scoring.
# Resumable: finished turns are marked on the GPU.
# Optional env overrides (e.g. phase-2 screening): PROMPTS (GPU prompt file), VMSRC (VM problem file for scoring).
set -e
TAG=$1; K=$2; SEED=$3; ADAPTER=${4:-}
cd "$(dirname "$0")"; mkdir -p "agent/$TAG"
P=/root/autodl-tmp/posttrain-project; R=$P/runs/stage4-phase1/agent-$TAG; E=$P/eval-baseline-20260930
VENV=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b/.venv
ENV="set -o pipefail; cd $P && export PATH=$VENV/bin:\$PATH PYTHONPATH=$P/src VLLM_USE_FLASHINFER_SAMPLER=0"
VMP=/root/agent-cpu-local-20260929; VR=$VMP/runs/stage4-phase1/agent-$TAG
PROMPTS=${PROMPTS:-data/kodcode-rl-v1-public/gpu-dev-prompts.jsonl}; VMSRC=${VMSRC:-cpu-data/kodcode-dev-v1.jsonl}; ST=$R/state.jsonl
retry(){ for i in 1 2 3 4 5 6; do "$@" && return 0; sleep 20; done; return 1; }
ssh gpu "mkdir -p $R"; ssh vm "mkdir -p $VR"
ssh gpu "test -f $ST" || ssh gpu "$ENV && python $E/agent_eval.py init --prompts $PROMPTS --k $K --out $ST"
for T in 1 2 3 4; do
  ssh gpu "test -f $R/t$T.applied" && continue
  if ! ssh gpu "test -f $R/t$T.generated"; then
    ssh gpu "$ENV && flock $P/runs/claude-diag-20260930/gpu.lock python $E/agent_eval.py gen --state $ST --turn $T --seed $SEED --requests $R/t$T-req.jsonl ${ADAPTER:+--adapter $ADAPTER} 2>$R/t$T-gen.log | tail -1 && touch $R/t$T.generated"
  fi
  retry scp -q gpu:$R/t$T-req.jsonl "agent/$TAG/"
  if [ "$(wc -l < "agent/$TAG/t$T-req.jsonl")" -gt 0 ]; then
    retry scp -q "agent/$TAG/t$T-req.jsonl" vm:$VR/
    ssh -o ServerAliveInterval=30 vm "cd $VMP && test -f $VR/t$T-res.jsonl || python3 claude_agent_tools_v2.py $VR/t$T-req.jsonl $VR/t$T-res.jsonl > $VR/t$T-tools.log 2>&1"
    retry scp -q vm:$VR/t$T-res.jsonl "agent/$TAG/"
  else
    : > "agent/$TAG/t$T-res.jsonl"
  fi
  retry scp -q "agent/$TAG/t$T-res.jsonl" gpu:$R/
  ssh gpu "$ENV && python $E/agent_eval.py apply --state $ST --results $R/t$T-res.jsonl 2>/dev/null | tail -1 && touch $R/t$T.applied"
done
ssh gpu "$ENV && python $E/agent_eval.py finalize --state $ST --prompts $PROMPTS --out $R/cands.jsonl 2>/dev/null | tail -1"
retry scp -q gpu:$R/cands.jsonl "agent/$TAG/"
retry scp -q "agent/$TAG/cands.jsonl" vm:$VR/
ssh -o ServerAliveInterval=30 vm "cd $VMP && python3 claude_score_v2.py $VR/cands.jsonl $VR/score --source $VMSRC --full > $VR/score.log 2>&1"
scp -rq vm:$VR/score "agent/$TAG/"
echo "agent-$TAG complete"
