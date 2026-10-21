"""Result figures (Figs. 5-11) generated from the per-seed JSON files and numbers.json."""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import cube, MAIN, ABL, EXPS

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#6b6a66", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False})
COL = {"HIABC-FDRL": "#2a78d6", "OBL-WHO-DDQN": "#eb6834", "GA-DQN": "#1baf7a", "DRL": "#c98500", "Auction": "#d55181",
       "IABC-CDRL": "#6250d6", "IABC(noQ)-FDRL": "#008300", "ABC-FDRL": "#e34948", "FDRL-fixed": "#7a7974"}
MK = {"HIABC-FDRL": "o", "OBL-WHO-DDQN": "s", "GA-DQN": "^", "DRL": "D", "Auction": "v", "IABC-CDRL": "P",
      "IABC(noQ)-FDRL": "X", "ABC-FDRL": "h", "FDRL-fixed": "<"}
LBL = {"Auction": "Multi-attribute auction"}
XL = {"compute_jobs": "Number of compute-intensive jobs", "data_cloudlets": "Number of data-intensive cloudlets",
      "data_size": "Total data size (GB)"}
logs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "seed_*.json")))]
N = json.load(open(os.path.join(RES, "numbers.json")))
n = len(logs)


def ci(a, axis=0):
    m = np.nanmean(a, axis=axis)
    h = 1.96 * np.nanstd(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])
    return m, h


def sweep_fig(exp, fname):
    cc = cube(logs, MAIN, "cost")[exp]
    cs = cube(logs, MAIN, "sla_violation")[exp]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.32))
    a, pts = cc
    for mi, m in enumerate(MAIN):
        mu, h = ci(a[:, mi, :])
        off = (mi - 2) * (pts[1] - pts[0]) * 0.03
        axs[0].errorbar(np.asarray(pts) + off, mu, yerr=h, color=COL[m], marker=MK[m], ms=5, lw=1.6, capsize=2.5,
                        label=LBL.get(m, m), zorder=3 if m == "HIABC-FDRL" else 2)
        mu2, h2 = ci(cs[0][:, mi, :] * 100)
        axs[1].errorbar(np.asarray(pts) + off, mu2, yerr=h2, color=COL[m], marker=MK[m], ms=5, lw=1.6, capsize=2.5)
    axs[0].set_xlabel(XL[exp]); axs[0].set_ylabel("Total execution cost ($10^{-3}$ USD)")
    axs[1].set_xlabel(XL[exp]); axs[1].set_ylabel("Deadline violation (%)")
    axs[1].set_ylim(bottom=0)
    axs[0].set_title("(a) Billed cost, mean ± 95% CI", fontsize=8.5, loc="left")
    axs[1].set_title("(b) Late VMs, mean ± 95% CI", fontsize=8.5, loc="left")
    for ax in axs:
        ax.set_xticks(pts)
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=7.8)
    fig.subplots_adjust(bottom=0.27, left=0.08, right=0.99, top=0.9)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def reduction_fig(fname, metric_key="main_cost", title="cost"):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    base = [m for m in MAIN if m != "HIABC-FDRL"]
    for ei, e in enumerate(EXPS):
        T = N[metric_key][e]["tests"]
        data = [np.asarray(T[b]["seed_reduction"]) for b in base]
        ax = axs[ei]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="#1b1b1a"))
        for patch, b in zip(bp["boxes"], base):
            patch.set_facecolor(COL[b]); patch.set_alpha(0.35); patch.set_edgecolor(COL[b])
        rng = np.random.default_rng(0)
        for i, (d, b) in enumerate(zip(data, base)):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, d.size), d, s=9, color=COL[b], zorder=3, lw=0)
            p = T[b]["p_holm"]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.text(i + 1, ax.get_ylim()[1] if False else np.nanmax(d) + 0.6, star, ha="center", fontsize=7.5)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(range(1, len(base) + 1))
        ax.set_xticklabels(["OBL-WHO\nDDQN", "GA-DQN", "DRL", "Auction"], fontsize=7.3)
        ax.set_title({"compute_jobs": "(a) Compute-intensive jobs", "data_cloudlets": "(b) Data-intensive cloudlets",
                      "data_size": "(c) Data-size sweep"}[e], fontsize=8.3, loc="left")
    axs[0].set_ylabel(f"Paired {title} reduction by\nHIABC-FDRL (%)")
    fig.subplots_adjust(bottom=0.2, left=0.1, right=0.99, top=0.88)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def ablation_fig(fname):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.35, width_ratios=[1.1, 1]))
    order = ["HIABC-FDRL", "IABC-CDRL", "IABC(noQ)-FDRL", "ABC-FDRL", "FDRL-fixed", "DRL"]
    ob = N["overall"]["objective"]
    for i, m in enumerate(order):
        s = ob[m]
        axs[0].errorbar(s["mean"], i, xerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]], fmt=MK[m],
                        color=COL[m], ms=6, capsize=3, lw=1.6)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
    axs[0].set_xlabel("Test objective per VM, Eq. (4) ($10^{-3}$ USD)")
    axs[0].set_title("(a) Ablation, mean ± 95% CI", fontsize=8.5, loc="left")
    tun = N["tuning"]
    for m in ["HIABC-FDRL", "IABC(noQ)-FDRL", "ABC-FDRL"]:
        mu = np.asarray(tun[m]["trace_mean"]); sd = np.asarray(tun[m]["trace_sd"])
        x = np.arange(1, mu.size + 1)
        h = 1.96 * sd / np.sqrt(n)
        lab = {"HIABC-FDRL": "Q-guided IABC", "IABC(noQ)-FDRL": "IABC without Q-table", "ABC-FDRL": "Standard ABC"}[m]
        axs[1].plot(x, mu, color=COL[m], lw=1.8, label=lab, marker=MK[m], ms=3.5, markevery=3)
        axs[1].fill_between(x, mu - h, mu + h, color=COL[m], alpha=0.12, lw=0)
    axs[1].set_xlabel("Objective evaluations"); axs[1].set_ylabel("Best validation objective per VM")
    axs[1].set_title("(b) Tuner convergence, mean ± 95% CI", fontsize=8.5, loc="left")
    axs[1].legend(fontsize=7.5)
    fig.subplots_adjust(bottom=0.17, left=0.17, right=0.99, top=0.9)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def training_fig(fname):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    grid = np.arange(20, 201, 5)
    for m in ["HIABC-FDRL", "FDRL-fixed", "IABC-CDRL", "DRL", "GA-DQN", "OBL-WHO-DDQN"]:
        curves = []
        for L in logs:
            T = L["training"][m]
            c = np.asarray(T["curve"])
            if "rounds" in T:
                eps = (np.arange(1, c.size + 1)) * 5 * T["local_episodes"]
            else:
                k = 20
                c = np.convolve(c, np.ones(k) / k, mode="valid")
                eps = np.arange(k, k + c.size)
            curves.append(np.interp(grid, eps, c))
        mu, h = ci(np.asarray(curves))
        ax.plot(grid, mu, color=COL[m], lw=1.7, label=m, marker=MK[m], ms=3.5, markevery=6)
        ax.fill_between(grid, mu - h, mu + h, color=COL[m], alpha=0.1, lw=0)
    ax.set_xlabel("Environment episodes consumed (all clients)")
    ax.set_ylabel("Training objective per VM\n(stochastic policy)")
    ax.legend(ncol=3, fontsize=7.5)
    fig.subplots_adjust(bottom=0.17, left=0.13, right=0.99, top=0.96)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def secondary_fig(fname):
    S = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "secondary_*.json")))]
    if not S:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.62))
    al = ["100.0", "1.0", "0.1"]
    lab = ["100\n(near IID)", "1.0\n(mild)", "0.1\n(strong)"]
    for j, (k, yl) in enumerate([("objective", "Test objective per VM"), ("pqa", "Price-quote accuracy (%)")]):
        v = np.array([[s["noniid"][a][k] for a in al] for s in S])
        mu, h = ci(v)
        axs[j].bar(range(3), mu, yerr=h, color=COL["HIABC-FDRL"], alpha=0.75, width=0.55, capsize=3,
                   error_kw=dict(lw=1, ecolor="#2b2b2a"))
        axs[j].set_xticks(range(3)); axs[j].set_xticklabels(lab, fontsize=7.4)
        axs[j].set_xlabel("Dirichlet alpha"); axs[j].set_ylabel(yl)
        lo = np.min(mu - h); hi = np.max(mu + h)
        axs[j].set_ylim(lo - 0.25 * (hi - lo) - 1e-9, hi + 0.25 * (hi - lo))
    axs[0].set_title("(a) Non-IID clients", fontsize=8.3, loc="left")
    axs[1].set_title("(b) Quote accuracy", fontsize=8.3, loc="left")
    Ks = ["5", "10", "15", "20"]
    v = np.array([[s["clients"][k]["comm_MB"] for k in Ks] for s in S])
    axs[2].plot([5, 10, 15, 20], v.mean(axis=0), color=COL["HIABC-FDRL"], marker="o", lw=1.8)
    axs[2].set_xticks([5, 10, 15, 20]); axs[2].set_xlabel("Federated clients K (R = 20)")
    axs[2].set_ylabel("Communication volume (MB)")
    axs[2].set_title("(c) Communication", fontsize=8.3, loc="left")
    fig.subplots_adjust(bottom=0.24, left=0.11, right=0.99, top=0.88)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sweep_fig("compute_jobs", "fig5_compute.png")
    sweep_fig("data_cloudlets", "fig6_cloudlets.png")
    sweep_fig("data_size", "fig7_datasize.png")
    reduction_fig("fig8_reductions.png", sys.argv[3] if len(sys.argv) > 3 else "main_cost",
                  "cost" if (len(sys.argv) <= 3 or "cost" in sys.argv[3]) else "objective")
    ablation_fig("fig9_ablation.png")
    training_fig("fig10_training.png")
    secondary_fig("fig11_scaling.png")
    print("figures written to", OUT)
