"""Writes mcmc_ergodic.tex: irreducible and reducible chains, a trapped and a mixing trace, and the running (ergodic) mean.

Run: python3 mcmc_ergodic.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "Ergodicity: every state can be reached from every other", 0, 8.7, 13.4, 13.4)
f.panel("b", "A chain that mixes and a chain that is trapped", 0, 3.2, 13.4, 8.5)
f.panel("c", "Only the mixing chain gives the right time average", 0, -2.0, 13.4, 3.0)


def graph(x0, y0, links, tag):
    pos = [(x0, y0), (x0 + 1.5, y0 + 0.5), (x0 + 3.0, y0), (x0 + 4.5, y0 + 0.5)]
    for k, (x, y) in enumerate(pos):
        f.add(f"  \\fill[cohesin!{15 + 10 * k}] ({x},{y}) circle (0.32);\\draw[cohesin,line width=1pt] ({x},{y}) circle (0.32);\n")
    for a, b in links:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ang = np.arctan2(yb - ya, xb - xa)
        f.add(f"  \\draw[ink!60,line width=1.3pt,{{Stealth[length=4pt]}}-{{Stealth[length=4pt]}}] ({xa + 0.36 * np.cos(ang):.2f},{ya + 0.36 * np.sin(ang):.2f})--({xb - 0.36 * np.cos(ang):.2f},{yb - 0.36 * np.sin(ang):.2f});\n")
    f.add(f"  \\node[lab] at ({x0 + 2.25},{y0 - 0.75}) {{{tag}}};\n")


graph(0.9, 11.3, [(0, 1), (1, 2), (2, 3)], "irreducible: ergodic")
graph(7.3, 11.3, [(0, 1), (2, 3)], "reducible: states 3, 4 are never reached from 1, 2")
f.add("  \\node[note,text width=11.8cm,anchor=north west] at (0.9,10.1) {\\textbf{Irreducible}: a path with non-zero probability connects any two states. \\textbf{Aperiodic}: the chain does not return only at multiples of some $k>1$. With both, the chain has one stationary distribution $\\pi$ and, for any start, $P(X_t=i)\\to\\pi_i$. This is \\textbf{ergodicity}.};\n")
# ---- trajectories on a double well
rng = np.random.default_rng(2)
def metro(T, n, step=0.25, x0=1.0):
    x = np.empty(n); x[0] = x0
    E = lambda u: (u * u - 1) ** 2 / 0.12
    for t in range(1, n):
        y = x[t - 1] + step * rng.normal()
        x[t] = y if rng.random() < np.exp(-(E(y) - E(x[t - 1])) / T) else x[t - 1]
    return x
n = 4000
trap, mix = metro(0.35, n, 0.12), metro(2.2, n, 0.5)
for k, (tr, name, col) in enumerate(((mix, "mixing chain (high $T$)", "cohesin"), (trap, "trapped chain (low $T$)", "ink!75"))):
    ax = Axes(f, 1.4 + 6.2 * k, 4.2, 4.6, 2.8, (0, n), (-1.7, 1.7), "step $t$" if k == 0 else "step $t$", "$x_t$" if k == 0 else "", xticks=(0, 2000, 4000), yticks=(-1, 0, 1) if k == 0 else ())
    ax.line(np.arange(n), tr, f"thin,{col}", 8)
    f.add(f"  \\node[lab] at ({1.4 + 6.2 * k + 2.3},7.55) {{{name}}};\n")
# ---- c: running mean
for tr, col, lab in ((mix, "cohesin", "mixing"), (trap, "ink!75", "trapped")):
    pass
rm = lambda x: np.cumsum(x) / np.arange(1, len(x) + 1)
ax = Axes(f, 1.6, -1.1, 6.0, 3.0, (0, n), (-1.2, 1.2), "step $t$", "running mean of $x$", xticks=(0, 2000, 4000), yticks=(-1, 0, 1))
ax.hline(0, "ln,dashed")
ax.line(np.arange(n), rm(mix), "line width=1.8pt,cohesin", 8)
ax.line(np.arange(n), rm(trap), "line width=1.8pt,ink!75", 8)
f.add("  \\node[note,text width=4.2cm,anchor=north west] at (8.6,2.0) {\\textcolor{cohesin}{Mixing}: the running mean $\\bar x_T=\\frac1T\\sum_t x_t$ goes to the ensemble average. Trapped (black): it stays near $+1$, and never finds the other well.};\n")
f.add("  \\node[lab,anchor=west] at (8.6,2.6) {$\\langle x\\rangle=0$ (ensemble average)};\n")
f.save(__file__)
