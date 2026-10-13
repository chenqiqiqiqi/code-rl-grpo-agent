#!/bin/bash
# Usage (on the VM, copy to /root/agent-cpu-local-20260929/): bash claude_start_scorer_stage4.sh ATTESTATION_JSON TAG
# Stage 4 phase 1 scoring gateway on 8080, loopback only. Same as claude_start_rl_services.sh except
# queue timeout 1000 s (32 concurrent requests; long full-scoring tails must not abort a run) and no
# tool service (single-turn only). The timeout changes only aborts, never a score.
set -euo pipefail
ATT=$1; TAG=$2
cd /root/agent-cpu-local-20260929
IMG=sha256:0106a6090782e770c43597f20b3e579584c575b1166d3a5dad969f18d4fe5e56
export PYTHONPATH=src
if pgrep -f 'posttrain[.]execution[.]service' >/dev/null; then echo 'a scoring gateway is already running'; exit 1; fi
setsid nohup python3 -m posttrain.execution.service --input cpu-data/train-reference-passed.jsonl \
  --extra-input cpu-data/dev-reference-passed.jsonl --image $IMG --attestation "$ATT" \
  --output runs/scorer-stage4-$TAG --workers 8 --queue-timeout 1000 --port 8080 \
  > runs/scorer-stage4-$TAG.log 2>&1 < /dev/null &
sleep 8
curl -s --noproxy '*' http://127.0.0.1:8080/identity; echo
