"""Statistical analysis of the per-seed result files; writes numbers.json used by the manuscript.

Unit of analysis: one independent seed (tuning + training + paired test workloads).
Per seed, metrics are averaged over the two test replicates of each point.
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "numbers.json")
MAIN = ["HIABC-FDRL", "OBL-WHO-DDQN", "GA-DQN", "DRL", "Auction"]
ABL = ["HIABC-FDRL", "IABC-CDRL", "IABC(noQ)-FDRL", "ABC-FDRL", "FDRL-fixed", "DRL"]
EXPS = ["compute_jobs", "data_cloudlets", "data_size"]
METRICS = ["cost", "sla_violation", "energy_kwh", "pqa", "utilisation", "mean_response", "objective", "price_cv"]


def load():
    files = sorted(glob.glob(os.path.join(RES, "seed_*.json")))
    return [json.load(open(f)) for f in files]


def cube(logs, methods, metric):
    """array [seed, method, experiment, point] averaged over reps."""
    pts = {e: sorted({r["x"] for r in logs[0]["results"] if r["experiment"] == e}) for e in EXPS}
    out = {}
    for e in EXPS:
        a = np.full((len(logs), len(methods), len(pts[e])), np.nan)
        for si, L in enumerate(logs):
            acc = {}
            for r in L["results"]:
                if r["experiment"] != e or r["method"] not in methods:
                    continue
                v = r[metric] / (r["x"] if (metric == "objective" and e != "data_size") else (120 if metric == "objective" else 1))
                acc.setdefault((r["method"], r["x"]), []).append(v)
            for (m, x), vs in acc.items():
                a[si, methods.index(m), pts[e].index(x)] = np.mean(vs)
        out[e] = (a, pts[e])
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    m = len(ps)
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rank_biserial(d):
    d = np.asarray(d); d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def summary(a):
    n = a.shape[0]
    m = float(np.mean(a)); sd = float(np.std(a, ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return dict(mean=m, sd=sd, ci_low=m - h, ci_high=m + h)


def compare(logs, methods, ref, metric="cost"):
    cu = cube(logs, methods, metric)
    res = {}
    for e in EXPS:
        a, pts = cu[e]
        seed_mean = np.nanmean(a, axis=2)                    # [seed, method]
        block = dict(points=pts, per_point={}, per_method={}, tests={})
        for mi, m in enumerate(methods):
            block["per_method"][m] = summary(seed_mean[:, mi])
            block["per_point"][m] = [summary(a[:, mi, k]) for k in range(len(pts))]
        # Friedman over methods (blocks = seeds)
        fr = stats.friedmanchisquare(*[seed_mean[:, i] for i in range(len(methods))])
        ranks = np.apply_along_axis(stats.rankdata, 1, seed_mean)
        block["friedman"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue),
                                 mean_rank={m: float(ranks[:, i].mean()) for i, m in enumerate(methods)})
        ri = methods.index(ref)
        ps, rows = [], []
        for mi, m in enumerate(methods):
            if m == ref:
                continue
            diff = seed_mean[:, mi] - seed_mean[:, ri]                      # positive = ref better
            red = 100.0 * diff / seed_mean[:, mi]
            w = stats.wilcoxon(seed_mean[:, ri], seed_mean[:, mi], alternative="two-sided")
            pt_red = 100.0 * (a[:, mi, :] - a[:, ri, :]) / a[:, mi, :]
            rows.append(dict(baseline=m, mean_reduction=float(red.mean()), red_ci=boot_ci(red, seed=mi),
                             min_point=float(np.nanmin(np.nanmean(pt_red, axis=0))),
                             max_point=float(np.nanmax(np.nanmean(pt_red, axis=0))),
                             wins=int((diff > 0).sum()), n=int(diff.size), W=float(w.statistic), p=float(w.pvalue),
                             r_rb=rank_biserial(diff), per_point_reduction=np.nanmean(pt_red, axis=0).tolist(),
                             seed_reduction=red.tolist()))
            ps.append(w.pvalue)
        for row, pa in zip(rows, holm(ps)):
            row["p_holm"] = pa
        block["tests"] = {r["baseline"]: r for r in rows}
        res[e] = block
    return res


def overall(logs, methods, metric):
    """mean over all 16 points per seed -> summary per method."""
    cu = cube(logs, methods, metric)
    allp = np.concatenate([cu[e][0] for e in EXPS], axis=2)
    sm = np.nanmean(allp, axis=2)
    return {m: summary(sm[:, i]) for i, m in enumerate(methods)}, sm


def main():
    logs = load()
    N = dict(n_seeds=len(logs))
    N["main_cost"] = compare(logs, MAIN, "HIABC-FDRL", "cost")
    N["main_objective"] = compare(logs, MAIN, "HIABC-FDRL", "objective")
    N["ablation_objective"] = compare(logs, ABL, "HIABC-FDRL", "objective")
    N["ablation_cost"] = compare(logs, ABL, "HIABC-FDRL", "cost")
    N["overall"] = {}
    for met in METRICS:
        s, sm = overall(logs, MAIN + [m for m in ABL if m not in MAIN], met)
        N["overall"][met] = s
    # overall paired tests across all 16 points for the main metric set
    allm = MAIN + [m for m in ABL if m not in MAIN]
    for met in ["cost", "objective", "sla_violation", "energy_kwh", "pqa"]:
        s, sm = overall(logs, allm, met)
        ri = allm.index("HIABC-FDRL")
        rows, ps = [], []
        for mi, m in enumerate(allm):
            if m == "HIABC-FDRL":
                continue
            d = sm[:, mi] - sm[:, ri]
            w = stats.wilcoxon(sm[:, ri], sm[:, mi]) if np.any(d != 0) else None
            rows.append(dict(baseline=m, mean_diff=float(d.mean()), rel=float(100 * d.mean() / sm[:, mi].mean()),
                             p=float(w.pvalue) if w else 1.0, r_rb=rank_biserial(d), wins=int((d > 0).sum())))
            ps.append(rows[-1]["p"])
        for r, pa in zip(rows, holm(ps)):
            r["p_holm"] = pa
        N.setdefault("overall_tests", {})[met] = {r["baseline"]: r for r in rows}
    # equivalence (two one-sided tests) of HIABC-FDRL against each learner, margin +/-5 %
    eq = {}
    for met in ["cost", "objective", "energy_kwh"]:
        s, sm = overall(logs, allm, met)
        ri = allm.index("HIABC-FDRL")
        eq[met] = {}
        for mi, m in enumerate(allm):
            if m in ("HIABC-FDRL", "Auction"):
                continue
            rel = 100.0 * (sm[:, ri] - sm[:, mi]) / sm[:, mi]          # positive: HIABC-FDRL higher
            nn = rel.size; mu = rel.mean(); se = rel.std(ddof=1) / np.sqrt(nn)
            p_low = 1 - stats.t.cdf((mu + 5.0) / se, nn - 1)
            p_up = stats.t.cdf((mu - 5.0) / se, nn - 1)
            lo, hi = mu - stats.t.ppf(0.95, nn - 1) * se, mu + stats.t.ppf(0.95, nn - 1) * se
            eq[met][m] = dict(mean_rel=float(mu), ci90=[float(lo), float(hi)], p_tost=float(max(p_low, p_up)),
                              equivalent=bool(max(p_low, p_up) < 0.05))
    N["equivalence"] = eq
    # tuning
    tun = {}
    for name in ["GA-DQN", "OBL-WHO-DDQN", "HIABC-FDRL", "ABC-FDRL", "IABC(noQ)-FDRL", "IABC-CDRL"]:
        bf = [L["tuning"][name]["best_f"] for L in logs]
        tr = np.array([L["tuning"][name]["trace"] for L in logs])
        ws = [L["tuning"][name]["wall_s"] for L in logs]
        tun[name] = dict(best_f=summary(np.asarray(bf)), trace_mean=tr.mean(axis=0).tolist(),
                         trace_sd=tr.std(axis=0, ddof=1).tolist(), wall_s=summary(np.asarray(ws)),
                         best_f_seeds=bf)
        hps = [L["tuning"][name]["hp"] for L in logs]
        tun[name]["hp_median"] = {k: float(np.median([h[k] for h in hps])) for k in hps[0]
                                  if isinstance(hps[0][k], (int, float))}
        if "target" in hps[0]:
            g = sum(1 for h in hps if h["target"] == "max")
            tun[name]["target_mode"] = f"greedy ({g}/{len(hps)})" if g >= len(hps) - g else f"expected ({len(hps) - g}/{len(hps)})"
        if "op_usage" in logs[0]["tuning"][name]:
            u = np.sum([L["tuning"][name]["op_usage"] for L in logs], axis=0)
            s_ = np.sum([L["tuning"][name]["op_success"] for L in logs], axis=0)
            tun[name]["op_usage"] = u.tolist(); tun[name]["op_success"] = s_.tolist()
    for a_, b_ in [("HIABC-FDRL", "ABC-FDRL"), ("HIABC-FDRL", "IABC(noQ)-FDRL")]:
        x = np.asarray(tun[a_]["best_f_seeds"]); y = np.asarray(tun[b_]["best_f_seeds"])
        w = stats.wilcoxon(x, y)
        tun[f"{a_}_vs_{b_}"] = dict(p=float(w.pvalue), wins=int((x < y).sum()), r_rb=rank_biserial(y - x))
    N["tuning"] = tun
    # training/federation/latency
    tr = {}
    for name in ["DRL", "HIABC-FDRL", "FDRL-fixed", "ABC-FDRL", "IABC(noQ)-FDRL", "IABC-CDRL", "GA-DQN", "OBL-WHO-DDQN"]:
        T = [L["training"][name] for L in logs]
        d = dict(train_s=summary(np.asarray([t["train_s"] for t in T])),
                 n_params=summary(np.asarray([t["n_params"] for t in T], float)),
                 curve_len=[len(t["curve"]) for t in T])
        if "comm_bytes" in T[0]:
            d["comm_MB"] = summary(np.asarray([t["comm_bytes"] / 1e6 for t in T]))
            d["rounds"] = summary(np.asarray([t["rounds"] for t in T], float))
            d["agg_ms"] = summary(np.asarray([np.mean(t["agg_ms"]) for t in T]))
            d["drift_last"] = summary(np.asarray([t["drift"][-1] for t in T]))
            d["local_episodes"] = [t["local_episodes"] for t in T]
        tr[name] = d
    N["training"] = tr
    lat = [L["latency"] for L in logs]
    N["latency"] = dict(decision_median_ms=summary(np.asarray([l["decision_ms"]["median"] for l in lat])),
                        decision_p95_ms=summary(np.asarray([l["decision_ms"]["p95"] for l in lat])),
                        decision_p99_ms=summary(np.asarray([l["decision_ms"]["p99"] for l in lat])),
                        decision_max_ms=float(np.max([l["decision_ms"]["max"] for l in lat])),
                        n_decisions=int(np.sum([l["decision_ms"]["n"] for l in lat])),
                        price_update_ms=summary(np.asarray([l["price_update_ms_median"] for l in lat])))
    N["wall_s"] = summary(np.asarray([L["wall_s"] for L in logs]))
    json.dump(N, open(OUT, "w"), indent=1)
    print("wrote", OUT, "seeds", len(logs))


if __name__ == "__main__":
    main()
