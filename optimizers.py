"""Gradient-free hyperparameter optimizers with an identical evaluation budget.

All optimizers minimise f(x) over the unit hypercube [0,1]^d; decoding to concrete
hyperparameters is done by the caller.  Every call to f counts against the budget.
"""
import numpy as np


class Budget:
    def __init__(self, f, budget):
        self.f, self.budget, self.n = f, budget, 0
        self.best_x, self.best_f = None, np.inf
        self.trace = []                     # best-so-far after each evaluation

    def __call__(self, x):
        if self.n >= self.budget:
            raise StopIteration
        y = float(self.f(np.clip(x, 0.0, 1.0)))
        self.n += 1
        if y < self.best_f:
            self.best_f, self.best_x = y, np.clip(x, 0.0, 1.0).copy()
        self.trace.append(self.best_f)
        return y


def _run(opt, f, d, budget, rng, **kw):
    B = Budget(f, budget)
    try:
        opt(B, d, rng, **kw)
    except StopIteration:
        pass
    return dict(best_x=B.best_x, best_f=B.best_f, trace=B.trace, evals=B.n, **getattr(B, "extra", {}))


# ----------------------------------------------------------------------------- ABC
def _abc(B, d, rng, sn=4, limit=4):
    X = rng.random((sn, d)); F = np.array([B(x) for x in X]); trial = np.zeros(sn)

    def try_neighbour(i):
        k = rng.choice([q for q in range(sn) if q != i]); j = rng.integers(d)
        v = X[i].copy(); v[j] = np.clip(v[j] + rng.uniform(-1, 1) * (v[j] - X[k, j]), 0, 1)   # Eq. (13)
        fv = B(v)
        if fv < F[i]:
            X[i], F[i], trial[i] = v, fv, 0
        else:
            trial[i] += 1
    while True:
        for i in range(sn):
            try_neighbour(i)
        fit = 1.0 / (1.0 + F - F.min())
        p = fit / fit.sum()
        for _ in range(sn):
            try_neighbour(int(rng.choice(sn, p=p)))
        for i in np.nonzero(trial > limit)[0]:
            X[i] = rng.random(d); F[i] = B(X[i]); trial[i] = 0


def abc(f, d, budget, rng):
    return _run(_abc, f, d, budget, rng)


# ---------------------------------------------------------------------- Q-guided IABC
def _iabc(B, d, rng, sn=4, limit=4, use_q=True, alpha_q=0.3, gamma_q=0.6, eps=0.2):
    n_ops = 4                      # 0 single-dim, 1 gbest-guided, 2 multi-dim, 3 opposition jump
    Q = np.zeros((3, n_ops))
    X = rng.random((sn, d)); F = np.array([B(x) for x in X]); trial = np.zeros(sn)
    omega = 1.0                    # adaptive neighbourhood radius
    hist = []
    usage = np.zeros(n_ops); success = np.zeros(n_ops)
    B.extra = dict(q_table=Q, op_usage=usage, op_success=success)

    def state(i):
        return 0 if trial[i] == 0 else (1 if trial[i] <= limit / 2 else 2)

    def candidate(i, op):
        k = rng.choice([q for q in range(sn) if q != i])
        v = X[i].copy()
        g = X[np.argmin(F)]
        if op == 0:
            j = rng.integers(d); v[j] += rng.uniform(-omega, omega) * (v[j] - X[k, j])
        elif op == 1:
            j = rng.integers(d)
            v[j] += rng.uniform(-omega, omega) * (v[j] - X[k, j]) + rng.uniform(0, 1.5) * (g[j] - v[j])
        elif op == 2:
            js = rng.choice(d, size=max(2, d // 2), replace=False)
            v[js] += rng.uniform(-omega, omega, js.size) * (v[js] - X[k, js])
        else:
            js = rng.choice(d, size=max(1, d // 3), replace=False)
            v[js] = 1.0 - v[js] + rng.normal(0, 0.05, js.size)       # opposition jump
        return np.clip(v, 0, 1)

    def step(i):
        nonlocal omega
        s = state(i)
        if use_q and rng.random() > eps:
            op = int(np.argmax(Q[s] + 1e-9 * rng.random(n_ops)))
        else:
            op = int(rng.integers(n_ops))
        v = candidate(i, op)
        fv = B(v)
        usage[op] += 1
        if fv < F[i]:
            r = 1.0 + min(1.0, (F[i] - fv) / (abs(F[i]) + 1e-12) * 10)
            X[i], F[i], trial[i] = v, fv, 0
            success[op] += 1
            hist.append(1)
        else:
            r = -1.0
            trial[i] += 1
            hist.append(0)
        s2 = state(i)
        Q[s, op] += alpha_q * (r + gamma_q * Q[s2].max() - Q[s, op])           # Eq. (14)
        if len(hist) >= 5:                                                     # 1/5-success rule
            rate = np.mean(hist[-5:])
            omega = float(np.clip(omega * (1.2 if rate > 0.2 else 0.85), 0.2, 1.5))
    while True:
        for i in range(sn):
            step(i)
        fit = 1.0 / (1.0 + F - F.min())
        bonus = np.array([max(Q[state(i)].max(), 0) for i in range(sn)])        # Q-guided onlookers
        p = fit * (1.0 + bonus) if use_q else fit
        p = p / p.sum()
        for _ in range(sn):
            step(int(rng.choice(sn, p=p)))
        best = int(np.argmin(F))
        for i in np.nonzero(trial > limit)[0]:
            if i == best:
                trial[i] = 0
                continue                                # best-so-far retention
            X[i] = np.clip(0.5 * (X[best] + rng.random(d)) + rng.normal(0, 0.1, d), 0, 1)
            F[i] = B(X[i]); trial[i] = 0


def iabc(f, d, budget, rng, use_q=True):
    return _run(_iabc, f, d, budget, rng, use_q=use_q)


# ------------------------------------------------------------------------------- GA
def _ga(B, d, rng, n=6, pm=None):
    pm = pm or 1.0 / d
    X = rng.random((n, d)); F = np.array([B(x) for x in X])
    while True:
        elite = int(np.argmin(F))
        newX = [X[elite].copy()]; newF = [F[elite]]
        while len(newX) < n:
            def tour():
                a, b = rng.integers(n, size=2)
                return X[a] if F[a] < F[b] else X[b]
            p1, p2 = tour(), tour()
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            span = hi - lo
            c = rng.uniform(lo - 0.5 * span, hi + 0.5 * span)        # BLX-0.5 crossover
            mut = rng.random(d) < pm
            c[mut] += rng.normal(0, 0.15, mut.sum())
            c = np.clip(c, 0, 1)
            newX.append(c); newF.append(B(c))
        X, F = np.array(newX), np.array(newF)


def ga(f, d, budget, rng):
    return _run(_ga, f, d, budget, rng)


# ------------------------------------------------------------------------- OBL-WHO
def _oblwho(B, d, rng, n=4, groups=2, pc=0.13, jump=0.3, max_it=8):
    X0 = rng.random((n, d))
    cand = np.vstack([X0, 1.0 - X0])                      # opposition-based initialisation
    Fc = np.array([B(x) for x in cand])
    order = np.argsort(Fc)[:n]
    X, F = cand[order].copy(), Fc[order].copy()
    it = 0
    while True:
        it += 1
        tdr = max(0.05, 1.0 - it / max_it)
        order = np.argsort(F)
        stallions = order[:groups]
        wh = X[order[0]].copy()
        for i in range(n):
            if i in stallions:
                z = (rng.random(d) < tdr) * rng.random(d) + (rng.random(d) >= tdr) * rng.random()
                r = rng.uniform(-2, 2)
                step = 2 * z * np.cos(2 * np.pi * r * z) * (wh - X[i])
                v = wh + step if rng.random() < 0.5 else wh - step          # leader move around the water hole
            else:
                s = X[stallions[i % groups]]
                if rng.random() > pc:
                    z = (rng.random(d) < tdr) * rng.random(d) + (rng.random(d) >= tdr) * rng.random()
                    r = rng.uniform(-2, 2)
                    v = 2 * z * np.cos(2 * np.pi * r * z) * (s - X[i]) + s       # grazing
                else:
                    other = X[stallions[(i + 1) % groups]]
                    v = 0.5 * (X[i] + other)                                     # mating
            v = np.clip(v, 0, 1)
            fv = B(v)
            if fv < F[i]:
                X[i], F[i] = v, fv
        if rng.random() < jump:                                                  # generation jumping
            lo, hi = X.min(axis=0), X.max(axis=0)
            w = int(np.argmax(F))
            o = np.clip(lo + hi - X[w], 0, 1)
            fo = B(o)
            if fo < F[w]:
                X[w], F[w] = o, fo


def oblwho(f, d, budget, rng):
    return _run(_oblwho, f, d, budget, rng)


OPTIMIZERS = dict(ga=ga, oblwho=oblwho, abc=abc, iabc=iabc,
                  iabc_noq=lambda f, d, b, r: iabc(f, d, b, r, use_q=False))
