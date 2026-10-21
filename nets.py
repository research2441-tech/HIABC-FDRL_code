"""Minimal NumPy multilayer perceptron with Adam, used by every learning agent."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / (a + c))
            if i == len(sizes) - 2:
                scale *= out_scale
            self.W.append(rng.normal(0.0, scale, (a, c)))
            self.b.append(np.zeros(c))
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        out = []
        for W, b in zip(self.W, self.b):
            out += [W, b]
        return out

    def n_params(self):
        return int(sum(p.size for p in self.params()))

    def get_flat(self):
        return np.concatenate([p.ravel() for p in self.params()])

    def set_flat(self, flat):
        i = 0
        for k in range(len(self.W)):
            n = self.W[k].size
            self.W[k] = flat[i:i + n].reshape(self.W[k].shape).copy(); i += n
            n = self.b[k].size
            self.b[k] = flat[i:i + n].copy(); i += n

    def forward(self, x, keep=False):
        acts = [x]
        h = x
        L = len(self.W)
        for k in range(L):
            h = h @ self.W[k] + self.b[k]
            if k < L - 1:
                h = np.tanh(h)
            acts.append(h)
        if keep:
            self._acts = acts
        return h

    def backward(self, dout):
        acts = self._acts
        L = len(self.W)
        grads = [None] * (2 * L)
        d = dout
        for k in reversed(range(L)):
            grads[2 * k] = acts[k].T @ d
            grads[2 * k + 1] = d.sum(axis=0)
            if k > 0:
                d = (d @ self.W[k].T) * (1.0 - acts[k] ** 2)
        return grads

    def adam_step(self, grads, lr, clip=0.5, b1=0.9, b2=0.999, eps=1e-8):
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if clip and gn > clip:
            grads = [g * (clip / gn) for g in grads]
        self.t += 1
        ps = self.params()
        for i, (p, g) in enumerate(zip(ps, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)
