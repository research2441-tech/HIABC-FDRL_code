"""Draws the four schematic figures (Figs. 1-4) of the manuscript with matplotlib."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

C = dict(env="#e8f1fb", envE="#2a78d6", acc="#fdeee6", accE="#d9541f", lrn="#e4f6ee", lrnE="#148a60",
         opt="#fff4d9", optE="#b97f00", srv="#efedfb", srvE="#6250d6", grey="#f3f3f1", greyE="#6b6a66",
         ink="#1b1b1a")


def box(ax, x, y, w, h, text, fc, ec, fs=8.3, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"],
            fontweight="bold" if bold else "normal", linespacing=1.25, wrap=False)


def arrow(ax, p, q, color="#4a4946", style="-|>", lw=1.1, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=1, shrinkB=1))


def canvas(w=7.2, h=4.6):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def label(ax, x, y, t, color, fs=8.6):
    ax.text(x, y, t, ha="left", va="bottom", fontsize=fs, color=color, fontweight="bold")


# ------------------------------------------------------------------ Fig. 1 architecture
def fig1():
    fig, ax = canvas(7.4, 5.0)
    for (y, h, fc, ec, t) in [(0.70, 0.27, C["env"], C["envE"], "Environment"),
                              (0.39, 0.27, C["acc"], C["accE"], "AU/WU pricing"),
                              (0.03, 0.32, C["lrn"], C["lrnE"], "Learning")]:
        ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                    fc=fc, ec=ec, lw=0.9, alpha=0.55))
        ax.text(0.038, y + h / 2, t, rotation=90, ha="center", va="center", fontsize=8.6, color=ec, fontweight="bold")
    box(ax, 0.07, 0.75, 0.19, 0.17, "Workload streams\nSPEC CPU2006-derived\nMalStone-derived", "white", C["envE"])
    box(ax, 0.31, 0.75, 0.38, 0.17, "Three provider domains (P1, P2, P3)\n10 hosts x 4 PEs each; VM admission,\nconsolidation (pack) or spreading", "white", C["envE"])
    box(ax, 0.74, 0.75, 0.21, 0.17, "Real-time electricity\nprice per provider\n(mean-reverting)", "white", C["envE"])
    box(ax, 0.07, 0.43, 0.19, 0.17, "Power sampling\n1 s, Eqs. (6)-(8)", "white", C["accE"])
    box(ax, 0.30, 0.43, 0.19, 0.17, "Static / dynamic split\nEqs. (9)-(10)", "white", C["accE"])
    box(ax, 0.56, 0.43, 0.19, 0.17, "Unit prices C_AU, C_WU\nEqs. (11)-(14)", "white", C["accE"])
    box(ax, 0.79, 0.43, 0.16, 0.17, "VM charge\nquote Eq. (3)\nbill Eq. (15)", "white", C["accE"])
    box(ax, 0.07, 0.07, 0.23, 0.20, "IABC tuner (offline)\nQ-guided operators,\nadaptive radius, scouts\nAlgorithm 1", "white", C["optE"])
    box(ax, 0.37, 0.07, 0.27, 0.20, "Federated server\nsample-weighted FedAvg\nEq. (21), R rounds", "white", C["srvE"])
    box(ax, 0.71, 0.07, 0.24, 0.20, "K client agents\nactor pi(a|s), critic Q(s,a)\nlocal replay buffer", "white", C["lrnE"])
    arrow(ax, (0.26, 0.835), (0.31, 0.835))
    arrow(ax, (0.74, 0.835), (0.69, 0.835))
    arrow(ax, (0.36, 0.75), (0.165, 0.60))
    arrow(ax, (0.26, 0.515), (0.30, 0.515)); arrow(ax, (0.49, 0.515), (0.56, 0.515)); arrow(ax, (0.75, 0.515), (0.79, 0.515))
    arrow(ax, (0.80, 0.75), (0.68, 0.60))
    arrow(ax, (0.30, 0.17), (0.37, 0.17)); ax.text(0.335, 0.19, "h*", ha="center", fontsize=8, color=C["optE"])
    arrow(ax, (0.64, 0.20), (0.71, 0.20)); arrow(ax, (0.71, 0.14), (0.64, 0.14))
    ax.text(0.675, 0.215, "theta_g", ha="center", fontsize=7.5); ax.text(0.675, 0.105, "theta_k", ha="center", fontsize=7.5)
    arrow(ax, (0.90, 0.43), (0.90, 0.27)); ax.text(0.89, 0.35, "state s_t,\nreward r_t", fontsize=7.5, va="center", ha="right")
    # action path routed through the gap between the pricing boxes
    ax.plot([0.75, 0.75, 0.525], [0.27, 0.355, 0.355], color=C["lrnE"], lw=1.2)
    arrow(ax, (0.525, 0.355), (0.525, 0.75), color=C["lrnE"], lw=1.2)
    ax.text(0.515, 0.315, "action a_t: provider and placement rule", fontsize=7.5, color=C["lrnE"], ha="right")
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2 pricing timeline
def fig2():
    fig, ax = canvas(7.4, 3.9)
    ax.text(0.03, 0.93, "One pricing interval (5 s) on provider p", fontsize=9, fontweight="bold")
    x0, x1 = 0.06, 0.94
    ax.plot([x0, x1], [0.74, 0.74], color="#4a4946", lw=1.2)
    for k in range(6):
        x = x0 + k * (x1 - x0) / 5
        ax.plot([x, x], [0.72, 0.76], color="#4a4946", lw=1)
        ax.text(x, 0.68, f"t+{k} s", ha="center", fontsize=7.5)
    for k in range(5):
        x = x0 + (k + 0.5) * (x1 - x0) / 5
        ax.text(x, 0.79, f"Pow_Host(t_{k+1})", ha="center", fontsize=7.3, color=C["accE"])
    box(ax, 0.04, 0.38, 0.27, 0.20, "E_DC(t) = sum of host energy\nE_st = idle power x active time\n+ boot energy\nE_Dy = E_DC - E_st", C["acc"], C["accE"], fs=7.8)
    box(ax, 0.365, 0.38, 0.27, 0.20, "AU-seconds of resident VMs\n-> C_AU = mu kappa_p E_st / AU\nexecuted MI (WU)\n-> C_WU = mu kappa_p E_Dy / WU", C["acc"], C["accE"], fs=7.8)
    box(ax, 0.69, 0.38, 0.27, 0.20, "Charge of VM j for interval:\nAU_j x resident s x C_AU\n+ WU_j x C_WU\n(sum over VMs = mu x energy cost)", C["acc"], C["accE"], fs=7.8)
    arrow(ax, (0.31, 0.48), (0.365, 0.48)); arrow(ax, (0.635, 0.48), (0.69, 0.48))
    arrow(ax, (0.5, 0.67), (0.5, 0.585))
    box(ax, 0.04, 0.06, 0.43, 0.20, "Admission of VM j at t0 (start of interval):\nquote = AU_j C_AU(t0) D_hat + WU_j C_WU(t0)\nD_hat from current CPU share and IO contention", C["env"], C["envE"], fs=7.8)
    box(ax, 0.53, 0.06, 0.43, 0.20, "Completion of VM j:\nbilled charge = sum of interval charges\nprice-quote accuracy PQA = 1 - |billed - quote| / billed", C["lrn"], C["lrnE"], fs=7.8)
    arrow(ax, (0.47, 0.16), (0.53, 0.16))
    arrow(ax, (0.825, 0.38), (0.825, 0.26))
    fig.savefig(os.path.join(OUT, "fig2_pricing_interval.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3 federated
def fig3():
    fig, ax = canvas(7.2, 4.2)
    box(ax, 0.22, 0.74, 0.56, 0.20, "Federated server\nweighted FedAvg (Eq. 21): theta_g = sum_k (n_k / n) theta_k\nbroadcast theta_g; payload bytes logged per round", C["srv"], C["srvE"], fs=8)
    xs = [0.02, 0.34, 0.69]
    names = ["Client 1", "Client 2", "Client K"]
    for x, nme in zip(xs, names):
        box(ax, x, 0.06, 0.29, 0.40,
            f"{nme}\nlocal workload mix\n(Dirichlet, alpha)\nsimulator replica, AU/WU billing\nlocal replay buffer\n(raw transitions stay here)\nE local episodes per round",
            C["lrn"], C["lrnE"], fs=7.3)
    ax.text(0.665, 0.26, "...", fontsize=14, ha="center", va="center", color="#4a4946")
    for x, sx in [(0.14, 0.29), (0.46, 0.47), (0.81, 0.68)]:
        arrow(ax, (sx, 0.74), (x, 0.46), color=C["srvE"])
        arrow(ax, (x + 0.05, 0.46), (sx + 0.05, 0.74), color=C["lrnE"], ls="--")
    ax.text(0.02, 0.60, "download theta_g\n(float32)", fontsize=7.6, color=C["srvE"])
    ax.text(0.86, 0.60, "upload\ntheta_k, n_k", fontsize=7.6, color=C["lrnE"])
    fig.savefig(os.path.join(OUT, "fig3_federated.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 4 IABC flowchart
def fig4():
    fig, ax = canvas(7.2, 5.6)
    W = 0.40
    steps = [
        (0.86, "Initialise SN food sources in [0,1]^d; decode to hyperparameters;\nevaluate f = validation objective after short FDRL training", C["opt"], C["optE"]),
        (0.71, "Employed phase: state s = stagnation level of the source;\nchoose operator a by epsilon-greedy Q(s, .)", C["opt"], C["optE"]),
        (0.56, "Apply operator: single-dimension Eq. (25), best-guided Eq. (26),\nmulti-dimension or opposition jump with radius omega", C["opt"], C["optE"]),
        (0.41, "Greedy replacement; reward r = +1 (+ gain) or -1;\nQ update Eq. (27); 1/5-success rule adapts omega, Eq. (28)", C["opt"], C["optE"]),
        (0.26, "Onlooker phase: roulette on fitness x (1 + max Q);\nrepeat operator selection and Q update", C["opt"], C["optE"]),
        (0.11, "Scout phase: trial > L -> reinitialise near the best source;\nbest-so-far source is never discarded", C["opt"], C["optE"]),
    ]
    for y, t, fc, ec in steps:
        box(ax, 0.08, y, 0.62, 0.10, t, fc, ec, fs=7.9)
    for (y1, _, _, _), (y2, _, _, _) in zip(steps[:-1], steps[1:]):
        arrow(ax, (0.39, y1), (0.39, y2 + 0.10))
    box(ax, 0.76, 0.43, 0.21, 0.18, "Budget of\nevaluations\nexhausted?", C["grey"], C["greyE"], fs=8)
    arrow(ax, (0.70, 0.16), (0.865, 0.43), rad=0.25)
    arrow(ax, (0.865, 0.61), (0.70, 0.76), rad=0.25)
    ax.text(0.765, 0.69, "no", fontsize=8.5, fontweight="bold")
    ax.text(0.93, 0.72, "yes", fontsize=8.5, fontweight="bold")
    box(ax, 0.76, 0.86, 0.21, 0.10, "return h*\nto FDRL", C["lrn"], C["lrnE"], fs=8)
    arrow(ax, (0.92, 0.61), (0.92, 0.86))
    fig.savefig(os.path.join(OUT, "fig4_iabc.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("diagrams written to", OUT)
