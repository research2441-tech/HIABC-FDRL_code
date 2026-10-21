"""Training loops: centralized DRL/DQN, federated actor-critic (FedAvg), evaluation."""
import time
import numpy as np
from sim import MultiCloudEnv, make_workload, N_ACTIONS, NP, PROVIDERS
from agents import DQNAgent, QActorCritic

DEFAULT_AC = dict(lr_actor=1e-3, lr_critic=1e-3, hidden=64, gamma=0.9, entropy=0.01, target="sarsa",
                  local_episodes=2, batch=64, utd=0.5, target_sync=200, reward_scale=0.1,
                  eps_start=0.5, eps_frac=0.5)
DEFAULT_DQN = dict(lr=1e-3, hidden=64, gamma=0.95, eps_frac=0.5, target_sync=200, batch=64,
                   reward_scale=0.1)


def sample_train_workload(rng, class_mix=None):
    n = int(rng.integers(100, 181))
    total = float(rng.uniform(1800, 4800))
    fam = "compute" if rng.random() < 0.5 else "data"
    return make_workload(rng, fam, n, total, class_mix=class_mix)


def validation_set(seed):
    rng = np.random.default_rng(10_000_000 + seed)
    return [make_workload(rng, "compute", 140), make_workload(rng, "compute", 180),
            make_workload(rng, "data", 140, 3000.0), make_workload(rng, "data", 120, 4200.0)]


def evaluate(policy, workloads, env=None):
    env = env or MultiCloudEnv()
    return [env.run(wl, policy)[0] for wl in workloads]


def train_central(kind, hp, seed, episodes):
    rng = np.random.default_rng(seed)
    env = MultiCloudEnv()
    hp = dict(hp, planned_episodes=episodes)
    if kind == "a2c":
        agent = QActorCritic(hp, rng)
    else:
        agent = DQNAgent(hp, rng, double=(kind == "ddqn"), total_steps=episodes * 140)
    curve = []
    for ep in range(episodes):
        wl = sample_train_workload(rng)
        m, traj = env.run(wl, agent.policy(greedy=False), record=True)
        agent.update(traj)
        curve.append(m["objective"] / len(wl["arrival"]))
    return agent, dict(curve=curve)


def dirichlet_mixes(K, alpha, rng):
    if alpha is None:
        return [np.full(4, 0.25) for _ in range(K)]
    return [rng.dirichlet(np.full(4, alpha)) for _ in range(K)]


def train_federated(hp, seed, episodes, K=5, alpha=1.0, rounds=None, mixes=None):
    """Sample-weighted FedAvg over K clients; each client trains E local episodes per round."""
    rng = np.random.default_rng(seed)
    E = int(hp.get("local_episodes", 2))
    R = rounds if rounds is not None else max(1, int(round(episodes / (K * E))))
    mixes = mixes if mixes is not None else dirichlet_mixes(K, alpha, np.random.default_rng(seed + 77))
    hp = dict(hp, planned_episodes=R * E)
    glob = QActorCritic(hp, np.random.default_rng(seed))
    clients = [QActorCritic(hp, np.random.default_rng(seed * 31 + k + 1)) for k in range(K)]
    envs = [MultiCloudEnv() for _ in range(K)]
    crngs = [np.random.default_rng(seed * 1000 + k) for k in range(K)]
    nbytes = glob.n_params() * 4                          # float32 payload per direction
    comm = 0
    agg_t, drift_r, curve = [], [], []
    for r in range(R):
        g = glob.get_flat()
        locals_, ns = [], []
        rc = []
        for k in range(K):
            c = clients[k]
            c.set_flat(g)
            comm += nbytes                                  # download
            n_k = 0
            for _ in range(E):
                wl = sample_train_workload(crngs[k], class_mix=mixes[k])
                m, traj = envs[k].run(wl, c.policy(greedy=False), record=True)
                n_k += c.update(traj)
                rc.append(m["objective"] / len(wl["arrival"]))
            locals_.append(c.get_flat())
            ns.append(max(n_k, 1))
            comm += nbytes                                  # upload
        t0 = time.perf_counter()
        w = np.asarray(ns, float) / np.sum(ns)
        newg = np.sum([wi * th for wi, th in zip(w, locals_)], axis=0)   # Eq. (16)
        agg_t.append(time.perf_counter() - t0)
        drift_r.append(float(np.mean([np.linalg.norm(th - newg) / np.sqrt(th.size) for th in locals_])))
        glob.set_flat(newg)
        curve.append(float(np.mean(rc)))
    return glob, dict(curve=curve, comm_bytes=comm, payload_bytes=nbytes, rounds=R, local_episodes=E,
                      agg_time=agg_t, drift=drift_r, K=K)


def val_objective(policy, val):
    ms = evaluate(policy, val)
    return float(np.mean([m["objective"] / len(w["arrival"]) for m, w in zip(ms, val)]))
