#!/usr/bin/env bash
# Usage (local Git Bash): bash stage4-phase1/tunnels_tool.sh   (run in background; kill to stop)
# Agent tool-service hops, separate from tunnels.sh so the scorer tunnel is never restarted:
# local 18081 -> VM 8081, and GPU 8081 -> local 18081.
OPTS="-o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -o BatchMode=yes"
( while true; do ssh -N $OPTS -L 127.0.0.1:18081:127.0.0.1:8081 vm; echo "$(date -Is) vm tool hop exited $?"; sleep 5; done ) &
( while true; do ssh -N $OPTS -R 127.0.0.1:8081:127.0.0.1:18081 gpu; echo "$(date -Is) gpu tool hop exited $?"; sleep 5; done ) &
wait
