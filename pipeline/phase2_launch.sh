#!/usr/bin/env bash
# Usage (local Git Bash, background): bash stage4-phase1/phase2_launch.sh
# Protocol v4 addenda 6/9/10: once the W@124 Agent screening is scored, merge the lenient rescue, select the
# phase-2 pool (single 0<p<1 OR Agent-lenient 0<p<1, minus the 98 decontamination exclusions), upload it,
# check the scorer/tool identities from the GPU, then start the C -> A training chain.
set -euo pipefail
cd "$(dirname "$0")/.."
A=stage4-phase1/agent/w124-shard2-agent; G=/root/autodl-tmp/posttrain-project
until grep -q "agent-w124-shard2-agent complete" stage4-phase1/agent/w124-shard2-agent.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') screening scored; merging lenient rescue"
.venv/Scripts/python.exe stage4-phase1/lenient_rescue.py merge $A/score $A/rescue-cands.jsonl $A/rescue-score $A/lenient
.venv/Scripts/python.exe scripts/select_kodcode_pool_phase2.py --single-records stage4-phase1/evals/w124-shard2-single/records \
  --agent-records $A/lenient/records --exclude reports/stage4-phase1/decontam-recheck/shard2_exclude_proposed.json \
  --output data/prepared/kodcode-rl-v3
scp -q -r data/prepared/kodcode-rl-v3 data/prepared/kodcode-rl-v3-source gpu:$G/data/prepared/
ssh gpu "cd $G && P=/root/autodl-tmp/posttrain-runtime/verl-1876b06d0a3e4e71e06230be10af14492ca8a75b/.venv/bin/python && \
  for x in 8080:gateway_identity 8081:tool_identity; do \
    curl -s --noproxy '*' --max-time 10 127.0.0.1:\${x%%:*}/identity | \$P -c \"import json,sys;a=json.load(sys.stdin);b=json.load(open('evidence/stage4-s4e/\${x##*:}.json'));assert a==b,'\${x}';print('\${x} ok')\"; done && \
  setsid -f bash stage4-phase1/phase2_train.sh evidence/stage4-s4e phase2c configs/single_rl_phase2c_v4.json data/prepared/kodcode-rl-v3 \
    phase2a configs/agent_rl_phase2a_v4.json data/prepared/kodcode-rl-v3 > runs/stage4-phase1/phase2_train.out 2>&1 < /dev/null"
echo "$(date '+%F %T') phase-2 training chain started"
