#!/usr/bin/env bash
# Usage (local Git Bash): bash stage4-phase1/tunnels.sh   (run in background; Ctrl-C / kill to stop)
# Keeps both hops of the scorer tunnel alive: local 18080 -> VM 8080, and GPU 8080 -> local 18080.
# A dropped hop still aborts the in-flight training step (by design); the loop only shortens downtime
# before a resume.
OPTS="-o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -o BatchMode=yes"
( while true; do ssh -N $OPTS -L 127.0.0.1:18080:127.0.0.1:8080 vm; echo "$(date -Is) vm hop exited $?"; sleep 5; done ) &
( while true; do ssh -N $OPTS -R 127.0.0.1:8080:127.0.0.1:18080 gpu; echo "$(date -Is) gpu hop exited $?"; sleep 5; done ) &
wait
