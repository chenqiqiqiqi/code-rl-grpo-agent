#!/usr/bin/env bash
# Usage (local Git Bash, background): bash stage4-phase1/probe_judge_loop.sh
# Serialises VM scoring for the probes: waits for the VM to be free, fetches each probe's
# held-out probe-set candidates from the GPU as soon as they exist, scores them, and runs the
# KodCode reference verification (backup plan) after probe1. Records are copied back locally.
cd "$(dirname "$0")/.."
VMD=/root/agent-cpu-local-20260929; G=/root/autodl-tmp/posttrain-project/runs/stage4-phase1
SSH="ssh -o ServerAliveInterval=30 -o ConnectTimeout=20"
vm_idle(){ until ! $SSH vm 'pgrep -f "claude_score_(trai[n]|v[2])" >/dev/null'; do sleep 30; done; }
for N in probe1 kodcode-ref probe2; do
  vm_idle
  if [ $N = kodcode-ref ]; then
    echo "$(date '+%T') verify KodCode references"
    $SSH vm "cd $VMD && python3 claude_score_v2.py claude-kodcode-reference-cands.jsonl runs/stage4-phase1/kodcode-ref --source cpu-data/kodcode-rl-v1-unverified.jsonl --full > runs/stage4-phase1/kodcode-ref.log 2>&1"
    $SSH vm "cd $VMD/runs/stage4-phase1 && tar -cf - kodcode-ref" | tar -C data/kodcode-rl-v1 -xf -
    continue
  fi
  until $SSH gpu "test -f $G/$N-probe-s42-cands.jsonl" || ! $SSH gpu 'pgrep -f "probe_chain[.]sh" >/dev/null'; do sleep 60; done
  $SSH gpu "test -f $G/$N-probe-s42-cands.jsonl" || { echo "$(date '+%T') $N candidates missing; chain ended"; exit 1; }
  scp -q gpu:$G/$N-probe-s42-cands.jsonl stage4-phase1/probe/ && scp -q stage4-phase1/probe/$N-probe-s42-cands.jsonl vm:$VMD/claude-stage4-$N-probe-s42-cands.jsonl
  echo "$(date '+%T') score $N probe set"
  $SSH vm "cd $VMD && python3 claude_score_train.py claude-stage4-$N-probe-s42-cands.jsonl runs/stage4-phase1/$N-probe-s42 > runs/stage4-phase1/$N-probe-s42.log 2>&1"
done
for N in m0 binary169 mixed90 probe1 probe2; do
  $SSH vm "cd $VMD/runs/stage4-phase1 && tar -cf - $N-probe-s42" | tar -C stage4-phase1/probe -xf -
done
echo "$(date '+%T') all probe scoring done"
