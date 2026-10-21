#!/usr/bin/env sh
# Main study: seeds 1..16 on two worker processes (odd / even seeds).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${1:-results}
mkdir -p "$OUT"
(for s in 1 3 5 7 9 11 13 15; do python3 experiments.py $s "$OUT"; done) > "$OUT/log_a.txt" 2>&1 &
(for s in 2 4 6 8 10 12 14 16; do python3 experiments.py $s "$OUT"; done) > "$OUT/log_b.txt" 2>&1 &
wait
