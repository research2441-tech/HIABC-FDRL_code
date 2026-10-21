# HIABC-FDRL: reproducibility package

Code for "Hybrid-Optimized Federated Deep Reinforcement Learning for Energy-Efficient Dynamic Pricing in
Heterogeneous Multi-Cloud Systems" (revised manuscript, Scientific Reports).

Every number, table and figure in the results section is regenerated from the raw per-seed JSON files that
this package writes. Nothing is hard-coded: the proposed method and every baseline are billed by the same
AU/WU pricing equations and differ only in the decisions they learn.

## Requirements

Python 3.10 or newer, NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). No GPU and no deep
learning framework are needed; the networks, Adam, replay buffers and federated averaging are written in
NumPy so that each update rule can be read directly.

## Files (single flat folder)

| File | Content |
|---|---|
| `sim.py` | Multi-cloud simulator: 3 domains x 10 hosts x 4 PEs, SPECpower-style linear host power, 1 s power sampling, 5 s pricing interval, AU/WU unit prices (Eqs. 6-15), quotes (Eq. 3), PQA (Eq. 16), mean-reverting tariffs (Eq. 1), SPEC CPU2006- and MalStone-derived workload generator |
| `nets.py` | NumPy MLP with Adam and gradient clipping |
| `agents.py` | Off-policy actor-critic used by DRL/FDRL (Eqs. 18-20), DQN and Double DQN |
| `train.py` | Centralised training, federated training with sample-weighted FedAvg (Eq. 21) and Dirichlet client mixtures (Eq. 22) |
| `optimizers.py` | GA, OBL-WHO, standard ABC, Q-guided IABC (Eqs. 24-28) and IABC without Q-table, all under the same evaluation budget |
| `baselines.py` | Multi-attribute auction pricing baseline |
| `experiments.py` | One complete seed: tuning, training, paired testing, latency |
| `secondary.py` | Non-IID (alpha = 100, 1, 0.1) and client-count (K = 5-20) experiments |
| `analyze.py` | Friedman, Wilcoxon signed-rank with Holm correction, rank-biserial effect size, bootstrap CIs; writes `numbers.json` |
| `make_figures.py`, `make_diagrams.py` | Figures 1-11 |
| `run_all.sh` | Runs the 16 main seeds on two worker processes |
| `after_main.sh` | Runs the secondary experiments, the analysis and the figures once the main seeds exist |
| `seed_01.json`-`seed_16.json`, `secondary_01.json`-`secondary_10.json`, `numbers.json` | Raw results and derived statistics used in the manuscript |

## Re-analyse the archived raw results

The raw files shipped in this folder are the ones used in the manuscript:

```bash
python analyze.py . numbers_check.json     # identical statistics to numbers.json
python make_figures.py . figures
```

## Reproduce from scratch

```bash
pip install -r requirements.txt
./run_all.sh results                     # 16 seeds, about 13 min per seed per core
for s in 1 2 3 4 5 6 7 8 9 10; do python secondary.py $s results; done
python analyze.py results results/numbers.json
python make_figures.py results figures
python make_diagrams.py figures
```

Set `OMP_NUM_THREADS=1` when running seeds in parallel. A quick smoke test with tiny budgets:

```bash
TUNE_EPISODES=4 TUNE_BUDGET=5 TRAIN_EPISODES=10 python experiments.py 99 smoke
```

Seeds fix the workload streams, tariff paths, network initialisation, client mixtures and tuner randomness, so a
rerun on the same software stack reproduces the JSON files. Wall-clock latencies depend on the machine.

## Licence

MIT (see `LICENSE`).
