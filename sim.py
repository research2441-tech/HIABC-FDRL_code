"""Multi-cloud simulator with AU/WU energy-proportional dynamic pricing.

Three provider domains, ten hosts per domain, four processing elements per host.
Power is sampled every second; AU and WU unit prices are recomputed every
pricing interval (5 s) from measured static and dynamic energy (Eqs. 3-11).
"""
import time
import numpy as np

SAMPLE_S = 1.0          # Time_samp in Eq. (4)
INTERVAL_S = 5          # pricing / scheduling interval
HOSTS_PER_PROVIDER = 10
PES = 4
VCPU_CAP = 8            # 2x CPU over-commitment limit per host
VM_SPEED = 0.4          # fraction of PE MIPS delivered to one vCPU
BOOT_S = 20.0           # boot energy expressed as seconds of idle power
MARGIN = 0.25           # provider margin applied to energy cost
COST_SCALE = (1.0 + MARGIN) / 3.6e6 * 1000.0   # J * USD/kWh -> 10^-3 USD
LATE_PENALTY = 0.25     # 10^-3 USD-equivalent penalty per deadline miss (reward only)
REF_VCPU, REF_MEM = 1.0, 1.7                  # m1.small reference VM (Eq. 1)

# Host power profiles: linear interpolation between SPECpower idle and peak values
PROVIDERS = [
    dict(name="P1 (EC2 profile)", pe_mips=3067.0, idle=58.4, peak=222.0, tariff=0.12, io=750.0, mem=24.0),
    dict(name="P2 (GoGrid profile)", pe_mips=2660.0, idle=93.7, peak=135.0, tariff=0.09, io=450.0, mem=16.0),
    dict(name="P3 (Rackspace profile)", pe_mips=2933.0, idle=41.6, peak=113.0, tariff=0.15, io=1200.0, mem=16.0),
]
NP = len(PROVIDERS)
NH = NP * HOSTS_PER_PROVIDER
H_PROV = np.repeat(np.arange(NP), HOSTS_PER_PROVIDER)
H_IDLE = np.array([PROVIDERS[p]["idle"] for p in H_PROV])
H_PEAK = np.array([PROVIDERS[p]["peak"] for p in H_PROV])
H_MIPS = np.array([PROVIDERS[p]["pe_mips"] for p in H_PROV]) * VM_SPEED
H_IO = np.array([PROVIDERS[p]["io"] for p in H_PROV])
H_MEM = np.array([PROVIDERS[p]["mem"] for p in H_PROV])
TARIFF = np.array([p["tariff"] for p in PROVIDERS])
VM_TYPES = {1: 1.7, 2: 3.75, 4: 7.5}          # vCPU -> memory (GB)
N_ACTIONS = NP * 2                              # provider x {pack, spread}
OBS_DIM = 8 + NP * 10
NOMINAL_MIPS = 1100.0
NOMINAL_IO = 750.0

# workload classes used for federated non-IID partitioning
CLASSES = ("compute-light", "compute-heavy", "data-light", "data-heavy")


TARIFF_THETA = 0.15     # mean reversion per pricing interval
TARIFF_SIGMA = 0.25     # innovation s.d. of the log-price per interval


def tariff_path(rng, n_intervals):
    """Provider-specific real-time electricity prices (USD/kWh): mean-reverting log process."""
    x = np.zeros((n_intervals, NP))
    x[0] = rng.normal(0.0, TARIFF_SIGMA / np.sqrt(2 * TARIFF_THETA - TARIFF_THETA ** 2), NP)
    for k in range(1, n_intervals):
        x[k] = (1 - TARIFF_THETA) * x[k - 1] + rng.normal(0.0, TARIFF_SIGMA, NP)
    return TARIFF[None, :] * np.exp(x)


def make_workload(rng, family, n_jobs, data_total_gb=3000.0, window_s=None, class_mix=None):
    """Generate one workload trace.

    family: 'compute' (SPEC CPU2006-derived) or 'data' (MalStone-derived).
    class_mix: optional probabilities over CLASSES; used for client-specific
    non-IID mixtures (the family is drawn from the mixture, then the class).
    """
    if class_mix is not None:
        cm = np.asarray(class_mix, dtype=float)
        p_data = cm[2] + cm[3]
        family = "data" if rng.random() < p_data else "compute"
        sub = cm[2:] if family == "data" else cm[:2]
        sub = sub / sub.sum() if sub.sum() > 0 else np.array([0.5, 0.5])
        cls = (2 if family == "data" else 0) + rng.choice(2, size=n_jobs, p=sub)
    elif family == "compute":
        cls = rng.choice(2, size=n_jobs)
    else:
        cls = 2 + rng.choice(2, size=n_jobs)
    if window_s is None:
        window_s = 60.0 if family == "compute" else 400.0
    arr = np.sort(rng.uniform(0.0, window_s, n_jobs))
    is_data = cls >= 2
    vcpu = np.where(is_data, rng.choice([1, 2], n_jobs, p=[0.6, 0.4]),
                    rng.choice([1, 2, 4], n_jobs, p=[0.5, 0.35, 0.15]))
    mem = np.array([VM_TYPES[int(v)] for v in vcpu])
    # compute-intensive length (MI per vCPU), log-uniform within the SPEC-derived range
    light = rng.uniform(np.log(3000), np.log(12000), n_jobs)
    heavy = rng.uniform(np.log(12000), np.log(38000), n_jobs)
    mi_c = np.exp(np.where(cls == 0, light, heavy))
    per = data_total_gb / max(n_jobs, 1)
    gb_d = np.where(cls == 2, rng.uniform(0.5, 1.0, n_jobs), rng.uniform(1.0, 1.5, n_jobs)) * per
    gb = np.where(is_data, gb_d, rng.uniform(0.1, 0.5, n_jobs))
    mi = np.where(is_data, 500.0 * gb, mi_c)
    nominal = np.where(is_data, np.maximum(gb * 1024 / NOMINAL_IO, mi / NOMINAL_MIPS), mi / NOMINAL_MIPS)
    slack = rng.uniform(1.6, 3.0, n_jobs)
    deadline = arr + slack * nominal + INTERVAL_S
    au = (vcpu / REF_VCPU) * (mem / REF_MEM)                  # Eq. (1)
    tariff = tariff_path(rng, 2400 // INTERVAL_S + 2)
    return dict(arrival=arr, is_data=is_data.astype(bool), vcpu=vcpu.astype(float), mem=mem,
                mi=mi, gb=gb, deadline=deadline, nominal=nominal, au=au, cls=cls,
                tariff=tariff)


class MultiCloudEnv:
    """Interval-driven simulator. The policy is called once per admitted VM."""

    def __init__(self, horizon_s=2400):
        self.horizon = horizon_s
        self.ref_cau = None

    # ------------------------------------------------------------------ state
    def _reset(self, wl):
        self.wl = wl
        n = len(wl["arrival"])
        self.n = n
        self.host = -np.ones(n, dtype=int)
        self.prog = np.zeros(n)
        self.done = np.zeros(n, dtype=bool)
        self.finish = np.full(n, np.nan)
        self.charge = np.zeros(n)
        self.quote = np.zeros(n)
        self.au_s = np.zeros(n)
        self.wu = np.zeros(n)
        self.h_vcpu = np.zeros(NH)
        self.h_mem = np.zeros(NH)
        self.h_ndata = np.zeros(NH)
        self.h_on = np.zeros(NH, dtype=bool)
        # unit prices, initialised from a nominal half-loaded host
        self.c_au = np.zeros(NP)
        self.c_wu = np.zeros(NP)
        for p in range(NP):
            pr = PROVIDERS[p]
            self.c_au[p] = COST_SCALE * pr["tariff"] * pr["idle"] / 8.0          # per AU-s
            self.c_wu[p] = COST_SCALE * pr["tariff"] * 0.5 * (pr["peak"] - pr["idle"]) / (2 * pr["pe_mips"] * VM_SPEED)
        self.c_au0 = self.c_au.copy()
        self.c_wu0 = self.c_wu.copy()
        self.energy_j = np.zeros(NP)
        self.static_j = np.zeros(NP)
        self.util_acc = 0.0
        self.util_n = 0
        self.unit_hist = []
        self.price_update_s = []
        self.kappa = self.wl["tariff"][0].copy()
        self.k_idx = 0

    def _candidate_hosts(self, j, p):
        """Return (pack_host, spread_host) for job j on provider p (-1 if infeasible)."""
        lo, hi = p * HOSTS_PER_PROVIDER, (p + 1) * HOSTS_PER_PROVIDER
        v, m = self.wl["vcpu"][j], self.wl["mem"][j]
        vc, mm, on = self.h_vcpu[lo:hi], self.h_mem[lo:hi], self.h_on[lo:hi]
        fits = (vc + v <= VCPU_CAP) & (mm + m <= H_MEM[lo:hi])
        on_fit = fits & on
        off_idx = np.nonzero(~on)[0]
        if on_fit.any():
            cand = np.nonzero(on_fit)[0]
            pack = lo + cand[np.argmax(vc[cand])]
        elif off_idx.size:
            pack = lo + off_idx[0]
        else:
            pack = -1
        nofold = on & (vc + v <= PES) & (mm + m <= H_MEM[lo:hi])
        if self.wl["is_data"][j]:
            nofold = nofold & (self.h_ndata[lo:hi] < 1)      # one IO-bound VM per host
        if nofold.any():
            cand = np.nonzero(nofold)[0]
            spread = lo + cand[np.argmin(vc[cand])]
        elif off_idx.size:
            spread = lo + off_idx[0]
        elif on_fit.any():
            cand = np.nonzero(on_fit)[0]
            key = self.h_ndata[lo:hi][cand] * 10 + vc[cand]
            spread = lo + cand[np.argmin(key)]
        else:
            spread = pack
        return pack, spread

    def _est_runtime(self, j, h):
        v = self.wl["vcpu"][j]
        share = min(1.0, PES / max(self.h_vcpu[h] + v, 1e-9))
        cpu_t = self.wl["mi"][j] / (H_MIPS[h] * share)
        if self.wl["is_data"][j]:
            io_t = self.wl["gb"][j] * 1024 / (H_IO[h] / (self.h_ndata[h] + 1))
            return max(cpu_t, io_t)
        return cpu_t

    def observe(self, j, now, queue_len):
        wl = self.wl
        f = [float(wl["is_data"][j]), wl["vcpu"][j] / 4, wl["mem"][j] / 7.5,
             np.log(wl["mi"][j]) / np.log(38000.0), wl["gb"][j] / 50.0,
             min(3.0, max(0.0, (wl["deadline"][j] - now) / wl["nominal"][j])) / 3.0,
             wl["au"][j] / 17.6, min(1.0, queue_len / 50.0)]
        mask = np.zeros(N_ACTIONS, dtype=bool)
        cands = []
        for p in range(NP):
            pack, spread = self._candidate_hosts(j, p)
            lo, hi = p * HOSTS_PER_PROVIDER, (p + 1) * HOSTS_PER_PROVIDER
            on = self.h_on[lo:hi]
            feas = pack >= 0
            mask[2 * p] = feas
            mask[2 * p + 1] = feas
            cands.append((pack, spread))
            mean_load = (self.h_vcpu[lo:hi][on].mean() / VCPU_CAP) if on.any() else 0.0
            f += [on.mean(), mean_load,
                  (self.h_vcpu[pack] + wl["vcpu"][j]) / VCPU_CAP if feas else 1.0,
                  float(feas and not self.h_on[pack]),
                  float(feas and not self.h_on[spread]),
                  (self.h_ndata[spread] / 4.0) if feas else 1.0,
                  min(3.0, self.c_au[p] / self.c_au0[p]) / 3.0,
                  min(3.0, self.c_wu[p] / self.c_wu0[p]) / 3.0,
                  self.kappa[p] / 0.15, float(feas)]
        return np.asarray(f, dtype=np.float64), mask, cands

    def _place(self, j, h):
        p = H_PROV[h]
        boot = 0.0
        if not self.h_on[h]:
            self.h_on[h] = True
            boot = H_IDLE[h] * BOOT_S
        D = self._est_runtime(j, h)
        self.quote[j] = self.wl["au"][j] * self.c_au[p] * D + self.wl["mi"][j] * self.wl["vcpu"][j] * self.c_wu[p]
        self.host[j] = h
        self.h_vcpu[h] += self.wl["vcpu"][j]
        self.h_mem[h] += self.wl["mem"][j]
        if self.wl["is_data"][j]:
            self.h_ndata[h] += 1
        return boot

    # ------------------------------------------------------------- episode
    def run(self, wl, policy, record=False):
        """policy(obs, mask, cands, j) -> action index. Returns metrics and trajectory."""
        self._reset(wl)
        n = self.n
        arr = wl["arrival"]
        nxt = 0
        queue = []
        traj = dict(obs=[], act=[], mask=[], rew=[], job=[]) if record else None
        accrued = 0.0
        t = 0
        lateness_pen = 0.0
        int_static = np.zeros(NP)
        int_energy = np.zeros(NP)
        int_au = np.zeros(NP)
        int_wu = np.zeros(NP)
        int_job_au = np.zeros(n)
        int_job_wu = np.zeros(n)
        n_dec = 0
        while t < self.horizon:
            if t % INTERVAL_S == 0:
                while nxt < n and arr[nxt] <= t:
                    queue.append(nxt)
                    nxt += 1
                waiting = []
                for qi, j in enumerate(queue):
                    obs, mask, cands = self.observe(j, t, len(queue) - qi)
                    if not mask.any():
                        waiting.append(j)
                        continue
                    a = policy(obs, mask, cands, j)
                    p, rule = divmod(int(a), 2)
                    h = cands[p][rule]
                    boot = self._place(j, h)
                    int_static[p] += boot
                    int_energy[p] += boot
                    if record:
                        if n_dec > 0:
                            traj["rew"].append(-accrued)
                            accrued = 0.0
                        traj["obs"].append(obs)
                        traj["act"].append(int(a))
                        traj["mask"].append(mask)
                        traj["job"].append(int(j))
                    n_dec += 1
                queue = waiting
            # ---------------- one sampling step (1 s)
            act = np.nonzero((self.host >= 0) & ~self.done)[0]
            if act.size:
                h = self.host[act]
                isd = wl["is_data"][act]
                vc = wl["vcpu"][act]
                ndata = np.bincount(h[isd], minlength=NH).astype(float)
                io_rate = np.where(isd, H_IO[h] / np.maximum(ndata[h], 1.0), 0.0)
                io_frac = np.where(isd, io_rate / (wl["gb"][act] * 1024.0), np.inf)
                need_mips = np.where(isd, io_frac * wl["mi"][act], np.inf)
                rho = np.where(isd, np.minimum(1.0, need_mips / H_MIPS[h]), 1.0)
                demand = vc * rho
                hd = np.bincount(h, weights=demand, minlength=NH)
                share_h = np.minimum(1.0, PES / np.maximum(hd, 1e-9))
                cpu_frac = H_MIPS[h] * share_h[h] / wl["mi"][act]
                frac = np.minimum(cpu_frac, io_frac)
                inc = np.minimum(frac * SAMPLE_S, 1.0 - self.prog[act])
                self.prog[act] += inc
                wu = inc * wl["mi"][act] * vc
                self.wu[act] += wu
                int_job_wu[act] += wu
                self.au_s[act] += SAMPLE_S
                int_job_au[act] += wl["au"][act] * SAMPLE_S
                u = np.minimum(1.0, hd / PES)
                on = self.h_on
                pw = np.where(on, H_IDLE + (H_PEAK - H_IDLE) * u, 0.0)
                e = np.bincount(H_PROV, weights=pw * SAMPLE_S, minlength=NP)
                s = np.bincount(H_PROV, weights=np.where(on, H_IDLE, 0.0) * SAMPLE_S, minlength=NP)
                int_energy += e
                int_static += s
                hp = H_PROV[h]
                int_au += np.bincount(hp, weights=wl["au"][act] * SAMPLE_S, minlength=NP)
                int_wu += np.bincount(hp, weights=wu, minlength=NP)
                if on.any():
                    self.util_acc += u[on].mean()
                    self.util_n += 1
                fin = act[self.prog[act] >= 1.0 - 1e-9]
                if fin.size:
                    self.done[fin] = True
                    self.finish[fin] = t + SAMPLE_S
                    late = (t + SAMPLE_S) > wl["deadline"][fin]
                    lateness_pen += LATE_PENALTY * late.sum()
                    accrued += LATE_PENALTY * late.sum()
                    for jj in fin:
                        hh = self.host[jj]
                        self.h_vcpu[hh] -= wl["vcpu"][jj]
                        self.h_mem[hh] -= wl["mem"][jj]
                        if wl["is_data"][jj]:
                            self.h_ndata[hh] -= 1
                        if self.h_vcpu[hh] <= 1e-9:
                            self.h_on[hh] = False
                            self.h_vcpu[hh] = 0.0
                            self.h_mem[hh] = 0.0
                            self.h_ndata[hh] = 0.0
            t += 1
            # ---------------- pricing interval close: Eqs. (3)-(11)
            if t % INTERVAL_S == 0:
                _tp = time.perf_counter()
                dyn = np.maximum(int_energy - int_static, 0.0)
                for p in range(NP):
                    if int_au[p] > 0:
                        self.c_au[p] = COST_SCALE * self.kappa[p] * int_static[p] / int_au[p]
                    if int_wu[p] > 0:
                        self.c_wu[p] = COST_SCALE * self.kappa[p] * dyn[p] / int_wu[p]
                jobs = np.nonzero(int_job_au > 0)[0]
                if jobs.size:
                    pj = H_PROV[self.host[jobs]]
                    ch = int_job_au[jobs] * self.c_au[pj] + int_job_wu[jobs] * self.c_wu[pj]
                    self.charge[jobs] += ch
                    accrued += ch.sum()
                # static energy without any AU (cannot happen except numerically) is still billed
                self.energy_j += int_energy
                self.static_j += int_static
                self.unit_hist.append((self.c_au.copy(), self.c_wu.copy()))
                self.cost_by_provider = getattr(self, "cost_by_provider", np.zeros(NP))
                self.k_idx = min(self.k_idx + 1, len(self.wl["tariff"]) - 1)
                self.kappa = self.wl["tariff"][self.k_idx].copy()
                int_static[:] = 0
                int_energy[:] = 0
                int_au[:] = 0
                int_wu[:] = 0
                int_job_au[:] = 0
                int_job_wu[:] = 0
                self.price_update_s.append(time.perf_counter() - _tp)
                if nxt >= n and not queue and self.done.all():
                    break
        unfinished = ~self.done
        late = np.where(self.done, self.finish > wl["deadline"], True)
        if record and n_dec > 0:
            # job-attributed reward: each admission decision receives the charge billed to
            # that VM plus the deadline penalty observed when the VM completes
            jb = np.asarray(traj["job"], dtype=int)
            traj["rew"] = list(-(self.charge[jb] + LATE_PENALTY * late[jb]))
        cost = float(self.charge.sum())
        q = self.quote[self.done]
        c = self.charge[self.done]
        pqa = float(100.0 * np.mean(np.clip(1.0 - np.abs(c - q) / np.maximum(c, 1e-12), 0.0, 1.0))) if c.size else 0.0
        resp = np.where(self.done, self.finish - wl["arrival"], np.nan)
        uh = np.array([np.concatenate(u) for u in self.unit_hist]) if self.unit_hist else np.zeros((1, 2 * NP))
        vol = float(np.mean(np.std(uh, axis=0) / (np.mean(uh, axis=0) + 1e-12))) if len(uh) > 1 else 0.0
        metrics = dict(cost=cost, price_cv=vol, energy_kwh=float(self.energy_j.sum() / 3.6e6),
                       sla_violation=float(late.mean()), pqa=pqa,
                       utilisation=float(self.util_acc / max(self.util_n, 1)),
                       mean_response=float(np.nanmean(resp)), unfinished=int(unfinished.sum()),
                       objective=cost + LATE_PENALTY * float(late.sum()), makespan=float(t))
        return metrics, traj
