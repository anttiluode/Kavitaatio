"""Negative control: a single-bit response dictionary need not decode joint wall memory.

All 256 binary codes are searched using linear superposition of one-bit response
changes. The *true* cavity is rerun for random multibit memories, which can
scatter repeatedly between altered wall patches. No training-on-test leakage.
"""
from __future__ import annotations
import numpy as np
from experiment import K, geometry, simulate


def evaluate(seed=555, trials=12):
    geom = geometry()
    base, delta = [], []
    eye = np.eye(K)
    for src in range(3):
        v0, _ = simulate(np.zeros(K), src, geom)
        base.append(v0[:, 0])
        delta.append(np.asarray([simulate(eye[k], src, geom)[0][:, 0] - v0[:, 0]
                                 for k in range(K)]).T)
    rng = np.random.default_rng(seed)
    hidden = rng.integers(0, 2, (trials, K))
    allcodes = ((np.arange(2**K)[:, None] >> np.arange(K)) & 1)
    out = {}
    for label, channels in (("single-source", [0]), ("three-source", [0,1,2])):
        y0 = np.concatenate([base[i] for i in channels])
        M = np.concatenate([delta[i] for i in channels], axis=0)
        templates = y0[None, :] + allcodes @ M.T
        hits = 0
        biterrs = []
        for bits in hidden:
            observed = np.concatenate([simulate(bits, i, geom)[0][:, 0] for i in channels])
            pred = allcodes[np.argmin(np.sum((templates - observed[None, :])**2, axis=1))]
            hits += int(np.array_equal(pred, bits))
            biterrs.append(int(np.count_nonzero(pred != bits)))
        out[label] = {"exact": hits, "trials": trials,
                      "mean_wrong_bits": float(np.mean(biterrs))}
    return out


if __name__ == "__main__":
    print(evaluate())
