"""Non-learning baseline: multi-attribute auction pricing."""
import numpy as np
from sim import NP, HOSTS_PER_PROVIDER, VCPU_CAP


def auction_policy(env, w_cost=0.5, w_time=0.35, w_rel=0.15):
    """Each feasible provider bids its consolidation (pack) host with an Eq. (2) quote and an
    expected completion time; the user selects the bid with the highest multi-attribute utility."""
    def f(obs, mask, cands, j):
        bids = []
        for p in range(NP):
            if not mask[2 * p]:
                continue
            h = cands[p][0]
            D = env._est_runtime(j, h)
            q = env.wl["au"][j] * env.c_au[p] * D + env.wl["mi"][j] * env.wl["vcpu"][j] * env.c_wu[p]
            lo, hi = p * HOSTS_PER_PROVIDER, (p + 1) * HOSTS_PER_PROVIDER
            load = env.h_vcpu[lo:hi].sum() / (VCPU_CAP * HOSTS_PER_PROVIDER)
            bids.append((p, q, D, load))
        qmin = min(b[1] for b in bids); tmin = min(b[2] for b in bids)
        best = max(bids, key=lambda b: w_cost * qmin / b[1] + w_time * tmin / b[2] + w_rel * (1 - b[3]))
        return 2 * best[0]
    return f
