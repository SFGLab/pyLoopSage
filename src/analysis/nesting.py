"""Nested loops: parents, children and nesting order.

A loop B is contained in loop A when A's left leg is at or left of B's and A's right leg is at or right of B's
(identical loops count once). The nesting order of a loop is the number of loops that contain it:

- order 0 = contained in no other loop: a parent if it holds loops, an orphan if it holds none (isolated loop)
- order k = child of order k (a chain parent > child 1 > ... > child k)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Nesting:
    valid: np.ndarray    # (K, T) loop counts (not a zero-length loop or a duplicate of another loop)
    order: np.ndarray    # (K, T) number of loops containing this loop
    length: np.ndarray   # (K, T) loop length in beads
    mid: np.ndarray      # (K, T) loop midpoint (beads)
    n_desc: np.ndarray   # (K, T) number of loops contained in this loop (children of any order)
    parent_len: np.ndarray  # (K, T) length of the smallest loop containing this one (NaN for parents)


def nesting(ms, ns, chunk: int = 400) -> Nesting:
    """Nesting of all loops in every snapshot; ms, ns are (K, T) leg positions."""
    lo, hi = np.minimum(ms, ns), np.maximum(ms, ns)
    K, T = lo.shape
    out = {k: np.zeros((K, T), t) for k, t in (("valid", bool), ("order", int), ("n_desc", int))}
    plen = np.full((K, T), np.nan)
    earlier = np.tri(K, k=-1, dtype=bool)[:, :, None]                       # j < i
    for a in range(0, T, chunk):
        l, h = lo[:, a:a + chunk], hi[:, a:a + chunk]
        same = (l[:, None] == l[None]) & (h[:, None] == h[None])           # (K, K, t): identical loops
        valid = ~(same & earlier).any(axis=1) & (h > l)                    # keep the first of identical loops
        length = h - l
        cont = ((l[:, None] <= l[None]) & (h[None] <= h[:, None]) & ~same) & valid[:, None]       # cont[j, i]: j contains i
        out["valid"][:, a:a + chunk] = valid
        out["order"][:, a:a + chunk] = cont.sum(axis=0)
        out["n_desc"][:, a:a + chunk] = (cont & valid[None]).sum(axis=1)
        p = np.where(cont, length[:, None], np.inf).min(axis=0)
        plen[:, a:a + chunk] = np.where(np.isinf(p), np.nan, p)
    return Nesting(out["valid"], out["order"], (hi - lo).astype(float), (lo + hi) / 2, out["n_desc"], plen)


def kind_of(order, n_desc, max_order: int = 3):
    """Class of a loop: 0 orphan, 1 parent, k + 1 child of order k (the last class is 'max_order or deeper')."""
    return np.where(order == 0, np.where(n_desc == 0, 0, 1), np.minimum(order, max_order) + 1)


def order_counts(nst: Nesting, max_order: int = 3) -> np.ndarray:
    """(max_order + 2, T): loops per snapshot of each class (orphans, parents, order 1, ..., 'max_order or deeper')."""
    k = kind_of(nst.order, nst.n_desc, max_order)
    return np.array([((k == c) & nst.valid).sum(axis=0) for c in range(max_order + 2)])


def production_nesting(res):
    """Nesting of every chain's post-burn-in snapshots, concatenated in time (loops stay (K, T) per chain)."""
    return [nesting(c.production(c.Ms), c.production(c.Ns)) for c in res.chains]


def pooled(parts, field):
    """Valid entries of a field over all chains as one flat array, together with the matching order array."""
    v = np.concatenate([getattr(p, field)[p.valid] for p in parts])
    o = np.concatenate([p.order[p.valid] for p in parts])
    return v, o


def nesting_summary(parts) -> dict:
    """Headline statistics over all post-burn-in snapshots."""
    order, _ = pooled(parts, "order")
    n_desc, _ = pooled(parts, "n_desc")
    orphan, parent = (order == 0) & (n_desc == 0), (order == 0) & (n_desc > 0)
    plen, o = pooled(parts, "parent_len")
    length, _ = pooled(parts, "length")
    kids = o > 0
    snap_max = np.concatenate([np.where(p.valid, p.order, 0).max(axis=0) for p in parts])
    n_par = np.concatenate([((p.order == 0) & (p.n_desc > 0) & p.valid).sum(axis=0) for p in parts])
    n_orph = np.concatenate([((p.order == 0) & (p.n_desc == 0) & p.valid).sum(axis=0) for p in parts])
    n_all = np.concatenate([p.valid.sum(axis=0) for p in parts])
    return dict(parent_fraction=float(parent.mean()), orphan_fraction=float(orphan.mean()), orphans_per_snapshot=float(n_orph.mean()), mean_order=float(order.mean()), mean_child_order=float(order[kids].mean()) if kids.any() else 0.0,
                max_order=int(order.max()), mean_max_order=float(snap_max.mean()), parents_per_snapshot=float(n_par.mean()),
                loops_per_snapshot=float(n_all.mean()), descendants_per_parent=float(n_desc[parent].mean()) if parent.any() else 0.0,
                length_ratio=float(np.mean(length[kids] / plen[kids])) if kids.any() else float("nan"),
                mean_length=[float(length[o == k].mean()) if (o == k).any() else float("nan") for k in range(4)])
