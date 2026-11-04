"""Anchor-level view of a BEDPE: position, strength, CTCF orientation and its confidence.

- Compartment (A/B) domains and their relation to loop anchors.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu

from preproc.readers import numeric_col, read_region_table, valid_prob


# ---- Anchor-level view of a BEDPE: position, strength, CTCF orientation and its confidence ----

CONF_MIN = 0.2          # below this an orientation call is treated as uncertain
CONFIGS = ("convergent", "tandem", "divergent", "uncertain")


@dataclass
class Anchors:
    bp: np.ndarray          # anchor centre (bp)
    strength: np.ndarray    # loop weight carried by the anchor
    p_rev: np.ndarray       # P(reverse), -1 = unknown
    loop: np.ndarray        # index of the loop each anchor belongs to
    left_bp: np.ndarray     # per loop: left / right anchor centre (bp)
    right_bp: np.ndarray
    loop_strength: np.ndarray
    config: np.ndarray      # per loop: one of CONFIGS

    @property
    def confidence(self) -> np.ndarray:
        return np.where(self.p_rev >= 0, np.abs(2 * self.p_rev - 1), 0.0)

    @property
    def orient(self) -> np.ndarray:
        """'forward' / 'reverse' / 'ambiguous' / 'unknown' per anchor."""
        o = np.where(self.p_rev < 0.5, "forward", "reverse").astype(object)
        o[self.confidence < CONF_MIN] = "ambiguous"
        o[self.p_rev < 0] = "unknown"
        return o

    @property
    def loop_len_bp(self) -> np.ndarray:
        return self.right_bp - self.left_bp


def _config(p_left, p_right) -> np.ndarray:
    """Loop configuration from the orientations of its left and right anchors."""
    sure = lambda p: (p >= 0) & (np.abs(2 * p - 1) >= CONF_MIN)
    ok = sure(p_left) & sure(p_right)
    lf, rf = p_left < 0.5, p_right < 0.5
    out = np.full(len(p_left), "uncertain", dtype=object)
    out[ok & lf & ~rf] = "convergent"          # left forward, right reverse: -> ... <-
    out[ok & ~lf & rf] = "divergent"
    out[ok & (lf == rf)] = "tandem"
    return out


def anchor_table(path, geom):
    """Anchors of every BEDPE loop in the region, or None if the file has no loops."""
    df = read_region_table(path, geom, paired=True)
    if df.empty:
        return None
    c1, c2 = (df[1] + df[2]).to_numpy() / 2, (df[4] + df[5]).to_numpy() / 2
    w = numeric_col(df, 6, 1.0)
    p1, p2 = numeric_col(df, 7, -1.0), numeric_col(df, 8, -1.0)
    if not (valid_prob(p1) and valid_prob(p2)):
        p1 = p2 = np.full(len(df), -1.0)
    swap = c1 > c2                                        # make anchor 1 the left one
    cl, cr = np.where(swap, c2, c1), np.where(swap, c1, c2)
    pl, pr = np.where(swap, p2, p1), np.where(swap, p1, p2)
    n = len(df)
    return Anchors(bp=np.concatenate([cl, cr]), strength=np.concatenate([w, w]), p_rev=np.concatenate([pl, pr]),
                   loop=np.concatenate([np.arange(n), np.arange(n)]), left_bp=cl, right_bp=cr,
                   loop_strength=w, config=_config(pl, pr))


def orientation_summary(a: Anchors) -> list:
    """Rows [class, anchors, share, median strength, mean confidence]."""
    rows, o = [], a.orient
    for name in ("forward", "reverse", "ambiguous", "unknown"):
        m = o == name
        rows.append([name, int(m.sum()), m.mean(), float(np.median(a.strength[m])) if m.any() else 0.0,
                     float(a.confidence[m].mean()) if m.any() else 0.0])
    return rows


def config_summary(a: Anchors) -> list:
    """Rows [configuration, loops, share, mean length (kb), mean strength]."""
    rows = []
    for name in CONFIGS:
        m = a.config == name
        rows.append([name, int(m.sum()), m.mean(), float(a.loop_len_bp[m].mean() / 1e3) if m.any() else 0.0,
                     float(a.loop_strength[m].mean()) if m.any() else 0.0])
    return rows


# ---- Compartment (A/B) domains and their relation to loop anchors ----------

def domains(comp, min_beads: int = 2):
    """Runs of equal sign: list of (first_bead, last_bead, 'A' or 'B'); runs shorter than min_beads are merged away."""
    sign = np.where(np.asarray(comp) >= 0, 1, -1)
    edges = np.flatnonzero(np.diff(sign)) + 1
    out = []
    for s, e in zip(np.r_[0, edges], np.r_[edges, len(sign)]):
        if e - s >= min_beads or not out:
            out.append([s, e - 1, "A" if sign[s] > 0 else "B"])
        else:
            out[-1][1] = e - 1                     # absorb a flicker into the previous domain
    merged = []
    for d in out:                                  # fuse neighbours of the same type
        if merged and merged[-1][2] == d[2]:
            merged[-1][1] = d[1]
        else:
            merged.append(d)
    return [tuple(d) for d in merged]


def compartment_summary(comp, anchor_strength, geom) -> dict:
    """Shares of A/B, domain sizes (kb) and anchor strength per compartment."""
    doms, bp = domains(comp), geom.bp_per_bead
    size = lambda t: np.array([(e - s + 1) * bp / 1e3 for s, e, k in doms if k == t])
    a_mask = np.asarray(comp) >= 0
    in_a, in_b = anchor_strength[a_mask], anchor_strength[~a_mask]
    p = mannwhitneyu(in_a, in_b).pvalue if len(in_a) > 2 and len(in_b) > 2 and anchor_strength.std() > 0 else np.nan
    return dict(frac_a=float(a_mask.mean()), n_domains=len(doms), size_a=size("A"), size_b=size("B"),
                anchors_a=in_a, anchors_b=in_b, p=float(p), domains=doms)
