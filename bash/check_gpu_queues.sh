#!/usr/bin/env bash
set -euo pipefail

# Queues we care about for standard CUDA / PyTorch work
QUEUES_REGEX='^(gpul40s|gpua40|gpua10|gpua100|gpuh100|gpuv100)$'

# Read bqueues output, keep header + selected queues
bqueues | awk -v queues_regex="$QUEUES_REGEX" '
NR==1 {
    printf "%-10s %-8s %-8s %-8s %-10s\n", "QUEUE", "PEND", "RUN", "NJOBS", "SCORE";
    next
}
$1 ~ queues_regex {
    queue=$1
    njobs=$8
    pend=$9
    run=$10

    score = pend * 100 + njobs

    printf "%-10s %-8s %-8s %-8s %-10s\n", queue, pend, run, njobs, score
}' | {
    read -r header
    echo "$header"
    tail -n +2 | sort -k5,5n
}
