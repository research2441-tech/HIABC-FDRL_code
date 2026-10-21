#!/usr/bin/env sh
# waits for the 16 main seeds, then runs the secondary experiments on seeds 1-10 and the analysis
R=${1:-results}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
until [ $(ls $R/seed_*.json 2>/dev/null | wc -l) -ge 16 ]; do sleep 20; done
(for s in 1 3 5 7 9; do python3 secondary.py $s $R; done) > $R/log_sec_a.txt 2>&1 &
(for s in 2 4 6 8 10; do python3 secondary.py $s $R; done) > $R/log_sec_b.txt 2>&1 &
wait
python3 analyze.py $R $R/numbers.json
python3 make_figures.py $R figures
python3 make_diagrams.py figures
echo PIPELINE_DONE
