"""Learning agents: off-policy actor-critic (DRL/FDRL core), DQN and Double DQN."""
import numpy as np
from nets import MLP
from sim import OBS_DIM, N_ACTIONS

NEG = -1e9


def masked_softmax(logits, mask):
    z = np.where(mask, logits, NEG)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(axis=-1, keepdims=True)


class DQNAgent:
    """Deep Q-network with experience replay; double=True gives Double DQN targets."""

    def __init__(self, hp, rng, double=False, total_steps=30000):
        self.hp = hp
        self.rng = rng
        self.double = double
        h = int(hp["hidden"])
        self.q = MLP([OBS_DIM, h, h, N_ACTIONS], rng)
        self.tq = MLP([OBS_DIM, h, h, N_ACTIONS], rng)
        self.tq.set_flat(self.q.get_flat())
        cap = 20000
        self.cap = cap
        self.buf_o = np.zeros((cap, OBS_DIM)); self.buf_a = np.zeros(cap, int)
        self.buf_r = np.zeros(cap); self.buf_o2 = np.zeros((cap, OBS_DIM))
        self.buf_m2 = np.zeros((cap, N_ACTIONS), bool); self.buf_d = np.zeros(cap)
        self.n = 0; self.ptr = 0
        self.steps = 0
        self.updates = 0
        self.total_steps = total_steps

    def eps(self):
        frac = min(1.0, self.steps / max(1.0, self.hp["eps_frac"] * self.total_steps))
        return 1.0 + frac * (0.05 - 1.0)

    def act(self, obs, mask, greedy=False):
        if not greedy and self.rng.random() < self.eps():
            return int(self.rng.choice(np.nonzero(mask)[0]))
        qv = self.q.forward(obs[None, :])[0]
        return int(np.argmax(np.where(mask, qv, NEG)))

    def policy(self, greedy=False):
        return lambda obs, mask, cands, j: self.act(obs, mask, greedy)

    def update(self, traj):
        hp = self.hp
        obs = np.asarray(traj["obs"]); T = len(obs)
        if T < 2:
            return 0
        act = np.asarray(traj["act"]); mask = np.asarray(traj["mask"])
        rew = np.asarray(traj["rew"]) / hp.get("reward_scale", 1.0)
        for t in range(T):
            i = self.ptr
            self.buf_o[i] = obs[t]; self.buf_a[i] = act[t]; self.buf_r[i] = rew[t]
            if t + 1 < T:
                self.buf_o2[i] = obs[t + 1]; self.buf_m2[i] = mask[t + 1]; self.buf_d[i] = 0.0
            else:
                self.buf_o2[i] = obs[t]; self.buf_m2[i] = mask[t]; self.buf_d[i] = 1.0
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
        self.steps += T
        if self.n < 500:
            return T
        bs = int(hp["batch"])
        gamma = hp["gamma"]
        for _ in range(max(1, int(T * hp.get("utd", 0.5)))):
            b = self.rng.integers(0, self.n, bs)
            q2t = self.tq.forward(self.buf_o2[b])
            if self.double:
                q2o = self.q.forward(self.buf_o2[b])
                a2 = np.argmax(np.where(self.buf_m2[b], q2o, NEG), axis=1)
                q2 = q2t[np.arange(bs), a2]
            else:
                q2 = np.where(self.buf_m2[b], q2t, NEG).max(axis=1)
            y = self.buf_r[b] + gamma * (1 - self.buf_d[b]) * q2
            qv = self.q.forward(self.buf_o[b], keep=True)
            err = qv[np.arange(bs), self.buf_a[b]] - y
            g = np.zeros_like(qv)
            g[np.arange(bs), self.buf_a[b]] = np.clip(err, -1, 1) / bs   # Huber gradient
            self.q.adam_step(self.q.backward(g), hp["lr"], clip=10.0)
            self.updates += 1
            if self.updates % int(hp["target_sync"]) == 0:
                self.tq.set_flat(self.q.get_flat())
        return T


class ReplayBuffer:
    def __init__(self, cap=20000):
        self.cap = cap
        self.o = np.zeros((cap, OBS_DIM)); self.a = np.zeros(cap, int); self.r = np.zeros(cap)
        self.m = np.zeros((cap, N_ACTIONS), bool)
        self.o2 = np.zeros((cap, OBS_DIM)); self.m2 = np.zeros((cap, N_ACTIONS), bool); self.d = np.zeros(cap)
        self.n = 0; self.ptr = 0

    def add_traj(self, obs, act, mask, rew):
        T = len(obs)
        for t in range(T):
            i = self.ptr
            self.o[i] = obs[t]; self.a[i] = act[t]; self.r[i] = rew[t]; self.m[i] = mask[t]
            if t + 1 < T:
                self.o2[i] = obs[t + 1]; self.m2[i] = mask[t + 1]; self.d[i] = 0.0
            else:
                self.o2[i] = obs[t]; self.m2[i] = mask[t]; self.d[i] = 1.0
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)


class QActorCritic:
    """Off-policy actor-critic for the discrete pricing/allocation action.

    The critic Q(s,a) is fitted to a temporal-difference target, either the expected value
    under the actor, r + gamma * sum_a' pi(a'|s') Q_tgt(s',a') ("sarsa"), or the greedy value
    r + gamma * max_a' Q_tgt(s',a') ("max"), using transitions sampled from the local replay
    buffer.  The actor is moved towards the Boltzmann policy softmax(Q(s,.)/alpha) of the critic
    (cross-entropy policy-improvement step).  Behaviour actions mix the actor with a decaying
    uniform exploration weight.  Raw transitions never leave the client.
    """

    def __init__(self, hp, rng):
        self.hp = hp
        self.rng = rng
        h = int(hp["hidden"])
        self.actor = MLP([OBS_DIM, h, h, N_ACTIONS], rng, out_scale=0.01)
        self.critic = MLP([OBS_DIM, h, h, N_ACTIONS], rng)
        self.tcritic = MLP([OBS_DIM, h, h, N_ACTIONS], rng)
        self.tcritic.set_flat(self.critic.get_flat())
        self.buf = ReplayBuffer()
        self.updates = 0
        self.steps = 0
        self.total_steps = 140 * int(hp.get("planned_episodes", 200))

    def eps(self):
        """Behaviour-policy mixing weight with a uniform feasible action (decays linearly)."""
        frac = min(1.0, self.steps / max(1.0, self.hp.get("eps_frac", 0.5) * self.total_steps))
        return self.hp.get("eps_start", 0.5) * (1.0 - frac) + 0.02 * frac

    def get_flat(self):
        return np.concatenate([self.actor.get_flat(), self.critic.get_flat()])

    def set_flat(self, flat):
        na = self.actor.n_params()
        self.actor.set_flat(flat[:na])
        self.critic.set_flat(flat[na:])
        self.tcritic.set_flat(flat[na:])

    def n_params(self):
        return self.actor.n_params() + self.critic.n_params()

    def act(self, obs, mask, greedy=False):
        logits = self.actor.forward(obs[None, :])[0]
        if greedy:
            return int(np.argmax(np.where(mask, logits, NEG)))
        if self.rng.random() < self.eps():
            return int(self.rng.choice(np.nonzero(mask)[0]))
        p = masked_softmax(logits[None, :], mask[None, :])[0]
        return int(self.rng.choice(N_ACTIONS, p=p))

    def policy(self, greedy=False):
        return lambda obs, mask, cands, j: self.act(obs, mask, greedy)

    def update(self, traj):
        hp = self.hp
        T = len(traj["obs"])
        self.steps += T
        if T < 2:
            return 0
        rew = np.asarray(traj["rew"]) / hp.get("reward_scale", 1.0)
        self.buf.add_traj(np.asarray(traj["obs"]), np.asarray(traj["act"]), np.asarray(traj["mask"]), rew)
        if self.buf.n < 256:
            return T
        bs = int(hp.get("batch", 64)); gamma = hp["gamma"]; alpha = hp["entropy"]
        B = self.buf
        n_upd = max(1, int(T * hp.get("utd", 0.5)))
        for _ in range(n_upd):
            b = self.rng.integers(0, B.n, bs)
            # ---- critic: expected-SARSA target under the current actor
            q2 = self.tcritic.forward(B.o2[b])
            if hp.get("target", "sarsa") == "max":
                v2 = np.where(B.m2[b], q2, NEG).max(axis=1)                 # greedy (Q-learning) target
            else:
                pi2 = masked_softmax(self.actor.forward(B.o2[b]), B.m2[b])
                v2 = (pi2 * np.where(B.m2[b], q2, 0.0)).sum(axis=1)       # expected-SARSA target
            y = B.r[b] + gamma * (1 - B.d[b]) * v2
            q = self.critic.forward(B.o[b], keep=True)
            err = q[np.arange(bs), B.a[b]] - y
            g = np.zeros_like(q)
            g[np.arange(bs), B.a[b]] = np.clip(err, -1, 1) / bs
            self.critic.adam_step(self.critic.backward(g), hp["lr_critic"], clip=10.0)
            # ---- actor: ascend E_pi[Q] + alpha*H
            m = B.m[b]
            qs = np.where(m, self.critic.forward(B.o[b]), 0.0)
            logits = self.actor.forward(B.o[b], keep=True)
            pi = masked_softmax(logits, m)
            if hp.get("actor_loss", "boltzmann") == "boltzmann":
                # policy improvement towards the Boltzmann policy of the critic,
                # pi_target(a|s) = softmax(Q(s,a)/alpha); cross-entropy gradient (pi - pi_target)
                tgt = masked_softmax(np.where(m, qs, NEG) / max(alpha, 1e-4), m)
                ga = (pi - tgt) * m / bs
            else:
                vbar = (pi * qs).sum(axis=1, keepdims=True)
                logp = np.log(pi + 1e-12) * m
                H = -(pi * logp).sum(axis=1, keepdims=True)
                ga = (-pi * (qs - vbar) + alpha * pi * (logp + H)) * m / bs
            self.actor.adam_step(self.actor.backward(ga), hp["lr_actor"], clip=1.0)
            self.updates += 1
            if self.updates % int(hp.get("target_sync", 200)) == 0:
                self.tcritic.set_flat(self.critic.get_flat())
        return T
