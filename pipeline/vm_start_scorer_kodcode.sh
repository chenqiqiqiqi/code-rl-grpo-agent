#!/bin/bash
# Usage (on the VM, copy to /root/agent-cpu-local-20260929/): bash claude_start_scorer_kodcode.sh ATTESTATION_JSON TAG
# Same as vm_start_scorer.sh plus the verified KodCode function tasks (cpu-data/kodcode-rl-v1.jsonl).
# Run as a script so `pgrep -f` patterns never match the caller's own command line.
set -euo pipefail
ATT=$1; TAG=$2
cd /root/agent-cpu-local-20260929
IMG=sha256:0106a6090782e770c43597f20b3e579584c575b1166d3a5dad969f18d4fe5e56
export PYTHONPATH=src
if pgrep -f 'posttrain[.]execution[.]service' >/dev/null; then echo 'a scoring gateway is already running'; exit 1; fi
setsid nohup python3 -m posttrain.execution.service --input cpu-data/train-reference-passed.jsonl \
  --extra-input cpu-data/dev-reference-passed.jsonl --extra-input cpu-data/kodcode-rl-v1.jsonl \
  --image $IMG --attestation "$ATT" --output runs/scorer-stage4-$TAG --workers 8 --queue-timeout 1000 --port 8080 \
  > runs/scorer-stage4-$TAG.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s --noproxy '*' http://127.0.0.1:8080/identity && break; sleep 2; done; echo
