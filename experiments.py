"""Per-seed experimental pipeline: tuning, training, paired testing, latency.

Usage:  python experiments.py SEED [OUT_DIR]
Writes OUT_DIR/seed_<SEED>.json with every raw measurement used in the manuscript.
"""
import json
import os
import sys
import time
import numpy as np

from sim import MultiCloudEnv, make_workload
from train import (DEFAULT_AC, DEFAULT_DQN, train_central, train_federated, validation_set,
                   val_objective)
from optimizers import OPTIMIZERS
from baselines import auction_policy

TUNE_EPISODES = int(os.environ.get("TUNE_EPISODES", 40))    # low-fidelity training per candidate
TUNE_BUDGET = int(os.environ.get("TUNE_BUDGET", 20))        # objective evaluations per optimizer
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", 200)) # final training budget for every learner
EXP_ID = {"compute_jobs": 1, "data_cloudlets": 2, "data_size": 3}
K_CLIENTS = 5
ALPHA = 1.0                 # Dirichlet concentration of client workload mixtures
HIDDEN = [32, 48, 64, 96, 128]


def logu(u, lo, hi):
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def decode_ac(x, federated=True):
    hp = dict(DEFAULT_AC)
    hp.update(lr_actor=logu(x[0], 1e-4, 3e-3), lr_critic=logu(x[1], 1e-4, 3e-3),
              hidden=HIDDEN[min(4, int(x[2] * 5))], gamma=0.5 + 0.49 * float(x[3]),
              entropy=logu(x[4], 1e-3, 3e-1), utd=0.25 + 0.75 * float(x[5]),
              target="max" if x[6] >= 0.5 else "sarsa")
    if federated:
        hp["local_episodes"] = [1, 2, 4][min(2, int(x[7] * 3))]
    return hp


def decode_dqn(x):
    hp = dict(DEFAULT_DQN)
    hp.update(lr=logu(x[0], 1e-4, 3e-3), hidden=HIDDEN[min(4, int(x[1] * 5))],
              gamma=0.5 + 0.49 * float(x[2]), eps_frac=0.2 + 0.6 * float(x[3]),
              target_sync=int(50 + 450 * float(x[4])), utd=0.25 + 0.75 * float(x[5]))
    return hp


def test_points():
    pts = []
    for n in (100, 120, 140, 160, 180):
        pts.append(("compute_jobs", n, dict(family="compute", n=n, data=3000.0)))
    for n in (100, 120, 140, 160, 180):
        pts.append(("data_cloudlets", n, dict(family="data", n=n, data=3000.0)))
    for d in (1800, 2400, 3000, 3600, 4200, 4800):
        pts.append(("data_size", d, dict(family="data", n=120, data=float(d))))
    return pts


def test_workloads(seed, reps=2):
    out = []
    for e, x, c in test_points():
        for r in range(reps):
            rng = np.random.default_rng([seed, EXP_ID[e], int(x), r, 424242])
            out.append((e, x, r, make_workload(rng, c["family"], c["n"], c["data"])))
    return out


def run_tests(name, policy_factory, tests, timing=False):
    rows = []
    lat = []
    for e, x, r, wl in tests:
        env = MultiCloudEnv()
        pol = policy_factory(env)
        if timing:
            def timed(obs, mask, cands, j, _p=pol):
                t0 = time.perf_counter(); a = _p(obs, mask, cands, j); lat.append(time.perf_counter() - t0)
                return a
            m, _ = env.run(wl, timed)
        else:
            m, _ = env.run(wl, pol)
        m.update(method=name, experiment=e, x=x, rep=r)
        if timing:
            m["price_update_ms_median"] = float(np.median(env.price_update_s) * 1e3)
        rows.append(m)
    return rows, lat


def tune(opt_name, kind, seed, federated):
    val = validation_set(seed)
    d = {("ac", True): 8, ("ac", False): 7}.get((kind, federated), 6)
    evals = []

    def f(x):
        if kind == "ac":
            hp = decode_ac(x, federated)
            if federated:
                ag, _ = train_federated(hp, seed * 7919 + len(evals), TUNE_EPISODES, K=K_CLIENTS, alpha=ALPHA)
            else:
                ag, _ = train_central("a2c", hp, seed * 7919 + len(evals), TUNE_EPISODES)
        else:
            hp = decode_dqn(x)
            ag, _ = train_central(kind, hp, seed * 7919 + len(evals), TUNE_EPISODES)
        y = val_objective(ag.policy(True), val)
        evals.append(y)
        return y
    t0 = time.perf_counter()
    res = OPTIMIZERS[opt_name](f, d, TUNE_BUDGET, np.random.default_rng(seed * 13 + len(opt_name)))
    res["wall_s"] = time.perf_counter() - t0
    hp = decode_ac(res["best_x"], federated) if kind == "ac" else decode_dqn(res["best_x"])
    out = dict(best_f=res["best_f"], trace=res["trace"], wall_s=res["wall_s"], hp=hp, evals=evals)
    for k in ("q_table", "op_usage", "op_success"):
        if k in res:
            out[k] = np.asarray(res[k]).tolist()
    return hp, out


def main(seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.perf_counter()
    tests = test_workloads(seed)
    log = dict(seed=seed, config=dict(TUNE_EPISODES=TUNE_EPISODES, TUNE_BUDGET=TUNE_BUDGET,
                                      TRAIN_EPISODES=TRAIN_EPISODES, K=K_CLIENTS, ALPHA=ALPHA),
               tuning={}, training={}, results=[], latency={})
    specs = [
        # name,              optimizer, kind,  federated
        ("GA-DQN",           "ga",      "dqn",  False),
        ("OBL-WHO-DDQN",     "oblwho",  "ddqn", False),
        ("DRL",              None,      "ac",   False),
        ("HIABC-FDRL",       "iabc",    "ac",   True),
        ("FDRL-fixed",       None,      "ac",   True),
        ("ABC-FDRL",         "abc",     "ac",   True),
        ("IABC(noQ)-FDRL",   "iabc_noq", "ac",  True),
        ("IABC-CDRL",        "iabc",    "ac",   False),
    ]
    for name, opt, kind, fed in specs:
        t0 = time.perf_counter()
        if opt is None:
            hp = dict(DEFAULT_AC)
        else:
            hp, tinfo = tune(opt, kind, seed, fed)
            log["tuning"][name] = tinfo
        t1 = time.perf_counter()
        tseed = seed * 100003 + 17
        if kind == "ac" and fed:
            agent, info = train_federated(hp, tseed, TRAIN_EPISODES, K=K_CLIENTS, alpha=ALPHA)
            info = dict(curve=info["curve"], comm_bytes=info["comm_bytes"], payload_bytes=info["payload_bytes"],
                        rounds=info["rounds"], local_episodes=info["local_episodes"],
                        agg_ms=[a * 1e3 for a in info["agg_time"]], drift=info["drift"])
        elif kind == "ac":
            agent, info = train_central("a2c", hp, tseed, TRAIN_EPISODES)
        else:
            agent, info = train_central(kind, hp, tseed, TRAIN_EPISODES)
        t2 = time.perf_counter()
        info.update(tune_s=t1 - t0, train_s=t2 - t1, hp=hp, n_params=int(agent.n_params()) if hasattr(agent, "n_params") else int(agent.q.n_params()))
        log["training"][name] = info
        rows, lat = run_tests(name, lambda env, a=agent: a.policy(True), tests, timing=(name == "HIABC-FDRL"))
        log["results"] += rows
        if lat:
            lat = np.asarray(lat) * 1e3
            log["latency"]["decision_ms"] = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)),
                                                 p99=float(np.percentile(lat, 99)), max=float(lat.max()), n=int(lat.size))
            log["latency"]["price_update_ms_median"] = float(np.median([r["price_update_ms_median"] for r in rows]))
        print(f"seed {seed} {name}: tune {t1-t0:.0f}s train {t2-t1:.0f}s "
              f"cost {np.mean([r['cost'] for r in rows]):.3f} sla {np.mean([r['sla_violation'] for r in rows]):.3f}",
              flush=True)
    rows, _ = run_tests("Auction", lambda env: auction_policy(env), tests)
    log["results"] += rows
    log["wall_s"] = time.perf_counter() - T0
    with open(os.path.join(out_dir, f"seed_{seed:02d}.json"), "w") as fh:
        json.dump(log, fh)
    print(f"seed {seed} done in {log['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "results")
