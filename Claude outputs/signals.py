"""Experimental signals (from loops/sites) and their simulated counterparts."""
from pathlib import Path

import numpy as np

from helpers.matrices import ps_curve
from preproc.readers import numeric_col


def detect_format(path) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".bedpe", ".bed", ".narrowpeak"):
        return ext[1:]
    raise ValueError(f"unsupported format '{ext}'")


def _beads(df, c1, c2, geom):
    return np.asarray(geom.bead_of(((df[c1] + df[c2]) // 2).to_numpy()))


def bedpe_pairs(df, geom):
    """(x, y, weight): anchor beads and weight of every loop."""
    return _beads(df, 1, 2, geom), _beads(df, 4, 5, geom), numeric_col(df, 6, 1.0)


def site_signal(mat, df, geom):
    """Per-bead site score (exp) and the bead's total simulated contacts (sim); BED/narrowPeak."""
    x, w = _beads(df, 1, 2, geom), numeric_col(df, 4, 1.0)
    exp, sim = np.zeros(geom.n_beads), np.zeros(geom.n_beads)
    np.add.at(exp, x, w)
    np.add.at(sim, x, mat[x].sum(axis=1) - mat[x, x])
    return exp, sim


def exp_ps(df, geom) -> np.ndarray:
    """Experimental P(s): mean loop weight per anchor distance."""
    x, y, w = bedpe_pairs(df, geom)
    d = np.abs(y - x)
    total, count = np.zeros(geom.n_beads), np.zeros(geom.n_beads)
    np.add.at(total, d, w); np.add.at(count, d, 1)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def oe_values(df, mat, geom):
    """Per-loop observed/expected: weight and simulated contact divided by their P(s)."""
    x, y, w = bedpe_pairs(df, geom)
    d = np.abs(y - x)
    sim = mat[x, y]
    return x, y, w / (exp_ps(df, geom)[d] + 1e-9), sim / (ps_curve(mat)[d] + 1e-9)


def aggregate_to_beads(x, y, exp_vals, sim_vals, n):
    """Mean (not sum) of per-loop values on each anchor bead."""
    out = []
    for vals in (exp_vals, sim_vals):
        tot, cnt = np.zeros(n), np.zeros(n)
        for b in (x, y):
            np.add.at(tot, b, vals); np.add.at(cnt, b, 1)
        out.append(np.divide(tot, cnt, out=np.zeros(n), where=cnt > 0))
    return out
