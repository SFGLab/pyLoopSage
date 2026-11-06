"""Writes mcmc_diag.tex: trace, ergodic mean and autocorrelation of the mean loop length in two real LoopSage chains.

Data: mcmc_demo.npz (two Metropolis chains, T = 1, chr1:78-86 Mb, 250 beads, 39 LEFs, 200 000 steps, one snapshot per 40 steps).
Run: python3 mcmc_diag.py
"""
from pathlib import Path

import numpy as np

from figlib import Axes, Fig

d = np.load(Path(__file__).with_name("mcmc_demo.npz"))
burn = int(d["burn"])
X = {k: d[f"len_{k}"] for k in "ab"}
n = len(X["a"])
f = Fig()
H = 4.0
f.panel("a", "Trace: the chain forgets its start, then fluctuates", 0, 8.2, 13.4, 12.2)
f.panel("b", "Ergodic mean: the running average after the burn-in", 0, 4.0, 13.4, 8.0)
f.panel("c", "Autocorrelation: how long the chain remembers", 0, -0.2, 13.4, 3.9)
COL = {"a": "cohesin", "b": "ink!75"}
# ---- a: traces
ax = Axes(f, 1.4, 9.3, 7.8, 2.3, (0, n), (4, 13), "snapshot", "mean loop length", xticks=(0, 1000, 2000, 3000, 4000, 5000), yticks=(5, 8, 11))
ax.fill([0, burn, burn, 0], [4, 4, 13, 13], "ink!10")
for k in "ab":
    ax.line(np.arange(n), X[k], f"line width=0.8pt,{COL[k]}", 5)
ax.label(burn + 330, 4.8, "burn-in", "note")
pool = np.concatenate([X[k][burn:] for k in "ab"]).mean()
ax.hline(pool, "ln,dashed")
f.add("  \\node[note,text width=3.4cm,anchor=north west] at (9.5,11.8) {Two chains from different random starts. Both start with short loops, rise, and then fluctuate around the same level (dashed). The grey part is the burn-in, which is removed.};\n")
# ---- b: ergodic mean
ax = Axes(f, 1.4, 4.95, 7.8, 2.3, (0, n - burn), (7, 10.5), "snapshots after the burn-in", "running mean", xticks=(0, 1000, 2000, 3000, 4000), yticks=(7, 8, 9, 10))
for k in "ab":
    y = X[k][burn:]
    ax.line(np.arange(len(y)), np.cumsum(y) / np.arange(1, len(y) + 1), f"line width=1.5pt,{COL[k]}", 5)
ax.hline(pool, "ln,dashed")
f.add("  \\node[note,text width=3.4cm,anchor=north west] at (9.5,6.85) {Early on the two chains differ; later they agree: a sign of convergence.};\n")
f.add("  \\node[lab,anchor=west] at (9.5,7.3) {$\\bar x_T=\\frac1T\\sum_{t=1}^{T}x_t\\to\\langle x\\rangle$};\n")
# ---- c: autocorrelation
def acf(x, m):
    y = x - x.mean(); a = np.correlate(y, y, "full")[len(y) - 1:]
    return (a / a[0])[:m]
LAG = 500
ax = Axes(f, 1.4, 0.7, 7.8, 2.2, (0, LAG), (-0.2, 1.0), "lag (snapshots)", "autocorrelation", xticks=(0, 100, 200, 300, 400, 500), yticks=(0, 0.5, 1.0))
ax.hline(0, "ln,dashed")
res = {}
for k in "ab":
    y = X[k][burn:]; a = acf(y, LAG)
    ax.line(np.arange(LAG), a, f"line width=1.6pt,{COL[k]}", 4)
    tau = 1.0
    for j in range(1, len(y) // 3):                          # integrated time, summed until the correlation is lost
        r = acf(y, j + 1)[j]
        if r < 0.05: break
        tau += 2 * r
    res[k] = (tau, len(y) / tau)
f.add("  \\node[note,text width=3.4cm,anchor=north west] at (9.5,2.85) {One independent sample per $\\tau_{\\mathrm{int}}$ snapshots. Here $\\tau_{\\mathrm{int}}\\approx %d$ and $%d$, so the 4750 snapshots of each chain are worth about $%d$ and $%d$ independent ones.};\n" % (round(res["a"][0], -1), round(res["b"][0], -1), round(res["a"][1]), round(res["b"][1])))
f.add("  \\node[lab,anchor=west] at (9.5,3.25) {$\\tau_{\\mathrm{int}}=1+2\\sum_{k\\ge1}\\rho(k)$};\n")
f.save(__file__)
print(res)
