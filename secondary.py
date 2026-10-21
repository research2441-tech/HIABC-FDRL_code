"""Non-IID robustness and client-count scaling for HIABC-FDRL.

Re-uses the IABC-selected hyperparameters of each seed (from results/seed_XX.json).
Usage: python secondary.py SEED RESULTS_DIR
"""
import json
import os
import sys
import time
import numpy as np

from train import train_federated, train_central
from experiments import test_workloads, run_tests

ALPHAS = [100.0, 1.0, 0.1]
CLIENTS = [5, 10, 15, 20]
ROUNDS = 20


def summarise(rows):
    return {k: float(np.mean([r[k] / (r["x"] if (k == "objective" and r["experiment"] != "data_size") else (120 if k == "objective" else 1))
                              for r in rows]))
            for k in ("cost", "objective", "sla_violation", "pqa", "energy_kwh")}


def main(seed, res):
    base = json.load(open(os.path.join(res, f"seed_{seed:02d}.json")))
    hp = base["training"]["HIABC-FDRL"]["hp"]
    tests = test_workloads(seed, reps=1)
    out = dict(seed=seed, hp=hp, noniid={}, clients={})
    for a in ALPHAS:
        t0 = time.perf_counter()
        ag, info = train_federated(hp, seed * 7 + 1, 200, K=5, alpha=a)
        rows, _ = run_tests("fed", lambda env, ag=ag: ag.policy(True), tests)
        s = summarise(rows)
        s.update(drift_last=float(np.mean(info["drift"][-3:])), curve=info["curve"],
                 curve_sd_last=float(np.std(info["curve"][-5:])), train_s=time.perf_counter() - t0)
        out["noniid"][str(a)] = s
        print(seed, "alpha", a, s["objective"], flush=True)
    hp1 = dict(hp); hp1["local_episodes"] = 1
    for K in CLIENTS:
        t0 = time.perf_counter()
        ag, info = train_federated(hp1, seed * 11 + K, 20 * K, K=K, alpha=1.0, rounds=ROUNDS)
        rows, _ = run_tests("fed", lambda env, ag=ag: ag.policy(True), tests)
        s = summarise(rows)
        s.update(comm_MB=info["comm_bytes"] / 1e6, payload_kB=info["payload_bytes"] / 1e3,
                 agg_ms=float(np.mean(info["agg_time"]) * 1e3), drift_last=float(np.mean(info["drift"][-3:])),
                 train_s=time.perf_counter() - t0, episodes=20 * K)
        out["clients"][str(K)] = s
        print(seed, "K", K, s["objective"], s["comm_MB"], flush=True)
    json.dump(out, open(os.path.join(res, f"secondary_{seed:02d}.json"), "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "results")
