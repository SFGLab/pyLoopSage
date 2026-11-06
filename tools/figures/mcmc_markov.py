"""Writes mcmc_markov.tex: a Markov chain, how its distribution converges to the target, and the Metropolis acceptance.

Run: python3 mcmc_markov.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "A Markov chain: the next state depends only on the present one", 0, 8.9, 13.4, 13.4)
f.panel("b", "The distribution of the chain approaches the target $\\pi$, whatever the start", 0, 3.7, 13.4, 8.7)
f.panel("c", "Detailed balance gives the Metropolis rule", 0, -1.7, 13.4, 3.5)
# ---- a: chain diagram
xs = [1.6, 4.0, 6.4]
for k, x in enumerate(xs):
    f.add(f"  \\fill[cohesin!{15 + 12 * k}] ({x},10.9) circle (0.45);\\draw[cohesin,line width=1pt] ({x},10.9) circle (0.45);\\node[lab] at ({x},10.9) {{{k + 1}}};\n")
for (a, b, up, lab) in ((0, 1, 1, "0.4"), (1, 0, -1, "0.3"), (1, 2, 1, "0.5"), (2, 1, -1, "0.6")):
    x0, x1 = xs[a], xs[b]; s = 1 if x1 > x0 else -1
    y = 10.9 + up * 0.18
    f.add(f"  \\draw[mv] ({x0 + s * 0.5:.2f},{y + up * 0.1:.2f}) .. controls ({(x0 + x1) / 2:.2f},{y + up * 0.75:.2f}) .. ({x1 - s * 0.5:.2f},{y + up * 0.1:.2f});\\node[note] at ({(x0 + x1) / 2:.2f},{y + up * 0.95:.2f}) {{{lab}}};\n")
f.add("  \\node[note] at (4.0,9.2) {arrows: transition probabilities $P_{ij}$ (the rest: stay)};\n")
# right: one path of the chain, and the Markov property as a formula
path = [1, 2, 3, 2, 2, 1]
for k, st in enumerate(path):
    x = 8.5 + 0.85 * k
    f.add(f"  \\fill[cohesin!{15 + 12 * (st - 1)}] ({x:.2f},11.5) circle (0.28);\\draw[cohesin,line width=0.9pt] ({x:.2f},11.5) circle (0.28);\\node[lab] at ({x:.2f},11.5) {{{st}}};\n")
    f.add(f"  \\node[note] at ({x:.2f},12.05) {{$X_{{{k}}}$}};\n")
    if k: f.add(f"  \\draw[mv] ({x - 0.85 + 0.3:.2f},11.5)--({x - 0.3:.2f},11.5);\n")
f.add("  \\node[note,anchor=west] at (8.2,10.7) {one path of the chain (a stochastic process)};\n")
f.add("  \\node[lab] at (10.6,9.9) {$P(X_{t+1}\\mid X_t,\\dots,X_0)=P(X_{t+1}\\mid X_t)$};\n")
f.add("  \\node[note] at (10.6,9.4) {Markov property: only the present state counts};\n")
# ---- b: convergence of the distribution (6-state Metropolis chain on an energy landscape)
n, T = 6, 1.0
E = np.array([0.8, 0.1, 1.2, 0.0, 0.9, 0.4])
pi = np.exp(-E / T); pi /= pi.sum()
P = np.zeros((n, n))
for i in range(n):
    for j in (i - 1, i + 1):
        if 0 <= j < n:
            P[i, j] = 0.5 * min(1, np.exp(-(E[j] - E[i]) / T))
    P[i, i] = 1 - P[i].sum()
for k, t in enumerate((0, 4, 60)):
    p = np.zeros(n); p[0] = 1
    p = p @ np.linalg.matrix_power(P, t)
    ax = Axes(f, 1.2 + 4.2 * k, 4.9, 3.4, 2.2, (-0.6, n - 0.4), (0, 1.0), "", "probability" if k == 0 else "", xticks=range(n), yticks=(0, 0.5, 1.0) if k == 0 else ())
    ax.bars(range(n), p, 0.7, "cohesin!70")
    ax.line(range(n), pi, "line width=1.4pt,ink,only marks", 1)
    for i in range(n):
        X, Y = ax.p(i, pi[i]); f.add(f"  \\fill[ink] ({X:.2f},{Y:.2f}) circle (0.08);\n")
    f.add(f"  \\node[lab] at ({1.2 + 4.2 * k + 1.7:.2f},7.55) {{after $t={t}$ steps}};\n")
f.add("  \\node[note,anchor=west] at (1.2,4.15) {horizontal axis: state. Bars: distribution of a chain started in state 0; black dots: target $\\pi_i\\propto e^{-E_i/T}$};\n")
# ---- c: acceptance
dE = np.linspace(-2, 4, 60)
ax = Axes(f, 1.6, -0.5, 4.2, 2.2, (-2, 4), (0, 1.1), "$\\Delta E=E_{\\mathrm{new}}-E_{\\mathrm{old}}$", "acceptance", xticks=(-2, 0, 2, 4), yticks=(0, 0.5, 1))
for T, c in ((0.5, "cohesin!45"), (1.0, "cohesin"), (2.0, "ink!75")):
    ax.line(dE, np.minimum(1, np.exp(-dE / T)), f"line width=1.8pt,{c}", 1)
f.add("  \\node[lab,anchor=west] at (6.8,2.5) {$\\pi_iP_{ij}=\\pi_jP_{ji}$};\\node[lab,anchor=west] at (6.8,1.8) {$P_{\\mathrm{accept}}=\\min\\!\\big(1,\\,e^{-\\Delta E/T}\\big)$};\\node[note,text width=6.0cm,anchor=north west] at (6.8,1.45) {Detailed balance holds for a symmetric proposal accepted with this probability.\\\\[3pt]\\textcolor{cohesin!55}{$T=0.5$}\\quad\\textcolor{cohesin}{$T=1$}\\quad $T=2$ (black)\\\\[3pt]Example: a move with $\\Delta E=1$ is accepted with probability $e^{-1}\\approx0.37$ at $T=1$, and $e^{-1/2}\\approx0.61$ at $T=2$.};\n")
f.save(__file__)
