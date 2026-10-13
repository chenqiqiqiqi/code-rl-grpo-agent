#!/usr/bin/env bash
# Usage (local Git Bash, background): bash stage4-phase1/vm_job_runner.sh JOBFILE
# Serial VM scoring queue. Each JOBFILE line: MODE NAME SOURCE
#   MODE  kdev (KodCode dev, full) | kprobe (KodCode probe set, full) | kshard2 (phase-2 shard-2 pool, full) | ktest (KodCode final test, full)
#         | tdev (TACO dev) | tprobe (TACO train probe set)
#   NAME  output name under runs/stage4-phase1 on the VM and stage4-phase1/evals locally
#   SOURCE gpu:<path relative to the GPU run dir> or local:<path>
# Lines are re-read every round, so jobs can be appended while it runs; finished NAMEs are skipped.
# Before each job: wait for the candidates, for no other VM scoring, and for no TACO training on the GPU
# (TACO rewards are timing-sensitive; KodCode training may overlap).
cd "$(dirname "$0")/.."
JOBS=$1; DONE=stage4-phase1/evals/done.txt; mkdir -p stage4-phase1/evals; touch $DONE
VMD=/root/agent-cpu-local-20260929; G=/root/autodl-tmp/posttrain-project/runs/stage4-phase1
SSH="ssh -o ServerAliveInterval=30 -o ConnectTimeout=20"
have(){ case $1 in gpu:*) $SSH gpu "test -f $G/${1#gpu:}";; local:*) test -f "${1#local:}";; esac; }
fetch(){ case $1 in gpu:*) scp -q "gpu:$G/${1#gpu:}" "$2";; local:*) cp "${1#local:}" "$2";; esac; }
taco_training(){ $SSH gpu 'ps -eo args | grep -E "code_rl.*single_rl_(tacosweep|binary_v4|mixed_v4)" | grep -v grep >/dev/null'; }
while true; do
  next=""
  while read -r MODE NAME SRC; do
    [ -z "$MODE" ] || [[ $MODE == \#* ]] && continue
    grep -qx "$NAME" $DONE && continue
    next="$MODE $NAME $SRC"; break
  done < "$JOBS"
  [ -z "$next" ] && { sleep 120; continue; }
  read -r MODE NAME SRC <<< "$next"; SRC=${SRC%$''}  # tolerate CRLF job files
  until have "$SRC"; do sleep 60; done
  fetch "$SRC" stage4-phase1/evals/$NAME-cands.jsonl && scp -q stage4-phase1/evals/$NAME-cands.jsonl vm:$VMD/claude-stage4-$NAME-cands.jsonl || { sleep 60; continue; }
  until ! $SSH vm 'pgrep -f "claude_score_(trai[n]|v[2])" >/dev/null' && ! taco_training; do sleep 60; done
  case $MODE in
    kdev)   CMD="python3 claude_score_v2.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME --source cpu-data/kodcode-dev-v1.jsonl --full";;
    kprobe) CMD="python3 claude_score_v2.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME --source cpu-data/kodcode-rl-v1.jsonl --full";;
    ktest)  CMD="python3 claude_score_v2.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME --source cpu-data/kodcode-test-v1.jsonl --full";;
    kshard2) CMD="python3 claude_score_v2.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME --source cpu-data/kodcode-shard2.jsonl --full";;
    tdev)   CMD="python3 claude_score_train.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME --dev";;
    tprobe) CMD="python3 claude_score_train.py claude-stage4-$NAME-cands.jsonl runs/stage4-phase1/$NAME";;
  esac
  echo "$(date '+%F %T') start $MODE $NAME"
  $SSH vm "cd $VMD && $CMD > runs/stage4-phase1/$NAME.log 2>&1"
  if $SSH vm "test -f $VMD/runs/stage4-phase1/$NAME/summary.json"; then
    $SSH vm "cd $VMD/runs/stage4-phase1 && tar -cf - $NAME" | tar -C stage4-phase1/evals -xf - 2>/dev/null
    echo "$NAME" >> $DONE; echo "$(date '+%F %T') done $NAME"
  else
    echo "$(date '+%F %T') $NAME did not finish; retrying later"; sleep 120
  fi
done
