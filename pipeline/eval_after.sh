#!/usr/bin/env bash
# Usage (local Git Bash, background): bash stage4-phase1/eval_after.sh NAME...
# For each KodCode RL run NAME: wait for the GPU chain's NAME.done, score its held-out KodCode probe
# candidates (full mode, function harness) and TACO development candidates on the VM (serialised with
# any other VM scoring), fetch the records and append paired comparisons vs M0 to stage4-phase1/kodcode/results.txt.
cd "$(dirname "$0")/.."
VMD=/root/agent-cpu-local-20260929; G=/root/autodl-tmp/posttrain-project/runs/stage4-phase1
SSH="ssh -o ServerAliveInterval=30 -o ConnectTimeout=20"
vm_idle(){ until ! $SSH vm 'pgrep -f "claude_score_(trai[n]|v[2])" >/dev/null'; do sleep 30; done; }
for NAME in "$@"; do
  until $SSH gpu "test -f $G/$NAME.done"; do sleep 60; done
  for TAG in kodcode-probe dev-s42; do
    scp -q gpu:$G/$NAME-$TAG-cands.jsonl stage4-phase1/kodcode/ && scp -q stage4-phase1/kodcode/$NAME-$TAG-cands.jsonl vm:$VMD/claude-stage4-$NAME-$TAG-cands.jsonl
  done
  vm_idle
  echo "$(date '+%T') score $NAME"
  $SSH vm "cd $VMD && python3 claude_score_v2.py claude-stage4-$NAME-kodcode-probe-cands.jsonl runs/stage4-phase1/$NAME-kodcode-probe --source cpu-data/kodcode-rl-v1.jsonl --full > runs/stage4-phase1/$NAME-kodcode-probe.log 2>&1"
  $SSH vm "cd $VMD/runs/stage4-phase1 && tar -cf - $NAME-kodcode-probe" | tar -C stage4-phase1/kodcode -xf - 2>/dev/null
  { echo "=== $NAME KodCode probe ($(date '+%F %T'))"; .venv/Scripts/python.exe stage4-phase1/compare_any.py stage4-phase1/kodcode/m0-kodcode-probe $NAME=stage4-phase1/kodcode/$NAME-kodcode-probe; } >> stage4-phase1/kodcode/results.txt 2>&1
  $SSH vm "cd $VMD && python3 claude_score_train.py claude-stage4-$NAME-dev-s42-cands.jsonl runs/stage4-phase1/$NAME-dev-s42 --dev > runs/stage4-phase1/$NAME-dev-s42.log 2>&1"
  $SSH vm "cd $VMD/runs/stage4-phase1 && tar -cf - $NAME-dev-s42" | tar -C stage4-phase1/dev/records -xf - 2>/dev/null
  { echo "=== $NAME TACO dev ($(date '+%F %T'))"; .venv/Scripts/python.exe stage4-phase1/compare_any.py eval-baseline-20260930/records/single-s42,eval-baseline-20260930/records/single-s43 $NAME=stage4-phase1/dev/records/$NAME-dev-s42; } >> stage4-phase1/kodcode/results.txt 2>&1
done
echo "$(date '+%T') eval_after done: $*"
