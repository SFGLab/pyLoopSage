"""Correlation statistics between an experimental and a simulated signal."""
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from helpers.stats import pearson_ci, strength_label


def permutation_p(exp, sim, n_perm: int = 500, seed: int = 0) -> float:
    """Empirical p-value of |Pearson r| by shuffling the simulated signal."""
    rng, r0 = np.random.default_rng(seed), pearsonr(exp, sim)[0]
    hits = sum(abs(pearsonr(exp, rng.permutation(sim))[0]) >= abs(r0) for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def correlate(exp, sim, mask_exp=None, mask_sim=None, drop_zero=True, n_perm=0) -> dict:
    """Pearson / Spearman / Kendall of two signals.

    Positions where BOTH raw signals are exactly zero are dropped (they agree
    trivially). `mask_*` are the raw (pre-normalisation) vectors used to find them.
    Returns None if fewer than 3 usable points remain.
    """
    exp, sim = np.asarray(exp, float), np.asarray(sim, float)
    n_total = len(exp)
    if drop_zero:
        me = exp if mask_exp is None else np.asarray(mask_exp, float)
        ms = sim if mask_sim is None else np.asarray(mask_sim, float)
        keep = ~((me == 0) & (ms == 0))
        exp, sim = exp[keep], sim[keep]
    n = len(exp)
    if n < 3 or np.std(exp) == 0 or np.std(sim) == 0:
        return None
    r, p = pearsonr(sim, exp)
    lo, hi = pearson_ci(r, n)
    return dict(n=n, n_total=n_total, pearson=float(r), p=float(p), ci_lo=lo, ci_hi=hi, r2=float(r * r),
                spearman=float(spearmanr(sim, exp)[0]), kendall=float(kendalltau(sim, exp)[0]),
                perm_p=permutation_p(exp, sim, n_perm) if n_perm else None,
                strength=strength_label(r), exp=exp, sim=sim)
