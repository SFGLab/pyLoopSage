"""Writes scope.tex: from a few rules to a contact map, and which patterns the rules can explain (toy data).

Run: python3 scope.py
"""
import numpy as np

from figlib import Fig

rng = np.random.default_rng(4)
N, ANCH = 30, [4, 14, 25]


def toy_map():
    """Average contact map of random loop states between three anchors (convergent CTCF)."""
    M = np.zeros((N, N))
    for _ in range(400):
        for _ in range(3):
            a, b = sorted(rng.choice(ANCH, 2, replace=False))
            m, n = a + rng.integers(-1, 2), b + rng.integers(-1, 2)
            M[m:n + 1, m:n + 1] += 0.06
            M[m, n] += 1.0
    i, j = np.indices(M.shape)
    M += 4.0 / (1 + np.abs(i - j))
    M = np.triu(M) + np.triu(M, 1).T
    return M / M.max()


def mini(f, x0, y0, s, M, cmap="cohesin"):
    """Heat map as small squares; M in [0, 1]."""
    n = len(M); c = s / n
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v > 0.04:
                f.add(f"  \\fill[{cmap}!{int(100 * min(v, 1) ** 0.6)}!white] ({x0 + c * j:.3f},{y0 + s - c * (i + 1):.3f}) rectangle +({c:.3f},{c:.3f});\n")
    f.add(f"  \\draw[ln] ({x0},{y0}) rectangle ({x0 + s},{y0 + s});\n")


def pattern(kind, n=18):
    M = 0.15 / (1 + np.abs(np.subtract.outer(range(n), range(n))))
    if kind == "tad":
        for a, b in ((1, 8), (9, 16)):
            M[a:b + 1, a:b + 1] += 0.55
    elif kind == "dot":
        M[3:13, 3:13] += 0.2; M[3, 12] = M[12, 3] = 1.0
    elif kind == "stripe":
        M[3, 3:] += 0.55; M[3:, 3] += 0.55
    else:
        t = (np.add.outer(np.arange(n) // 4, np.arange(n) // 4) % 2 == 0).astype(float) * 0.55
        M += t
    return np.clip(M, 0, 1)


f = Fig()
f.panel("a", "From a few rules to a contact map", 0, 4.4, 13.4, 9.9)
f.panel("b", "Which patterns can the rules explain?", 0, -1.2, 13.4, 4.2)
# ---- a: rules -> ensemble -> map
f.add("  \\node[box,minimum width=3.4cm,minimum height=2.6cm] (r) at (2.2,6.4) {\\bfseries Rules\\\\[3pt]{\\footnotesize a ring loads on the fibre}\\\\{\\footnotesize it extrudes a loop}\\\\{\\footnotesize CTCF can stop it}\\\\{\\footnotesize rings do not cross}};\n")
for k in range(3):
    y = 7.55 - 0.85 * k
    f.add(f"  \\draw[ln] (5.2,{y})--(8.0,{y});\n")
    loops = [[(0.2, 1.4)], [(0.6, 2.2), (1.0, 1.7)], [(1.4, 2.5), (0.2, 0.9)]][k]
    for a, b in loops:
        h = 0.2 * (b - a)
        f.add(f"  \\draw[cohesin,thin] ({5.2 + a},{y}) .. controls ({5.2 + a},{y + h * 1.6:.2f}) and ({5.2 + b},{y + h * 1.6:.2f}) .. ({5.2 + b},{y});\n")
f.text(6.6, 8.75, "{\\bfseries many random loop states}")
f.add("  \\draw[mv] (3.95,6.4)--(4.95,6.4);\\node[note] at (4.45,6.0) {sample};\n  \\draw[mv] (8.25,6.4)--(9.15,6.4);\\node[note] at (8.7,6.0) {average};\n")
mini(f, 9.4, 4.7, 3.4, toy_map())
f.text(11.1, 8.5, "{\\bfseries contact map}")
f.add("  \\node[note,text width=7cm,align=center] at (4.6,4.78) {nothing is drawn in by hand: TADs, dots and stripes are what the rules produce};\n")
# ---- b: patterns
cards = [("tad", "TAD", "loops that grow between two CTCF sites"), ("dot", "loop dot", "ring stalled at two convergent CTCF"), ("stripe", "stripe", "one end stays at CTCF, the other keeps extruding"), ("comp", "A/B compartments", "not from extrusion alone: needs attraction between like chromatin")]
for k, (kind, name, why) in enumerate(cards):
    x0 = 0.5 + 3.2 * k
    mini(f, x0 + 0.4, 0.75, 1.9, pattern(kind), "cohesin" if kind != "comp" else "B")
    f.text(x0 + 1.35, 2.85, "{\\bfseries " + name + "}", "lab")
    f.add(f"  \\node[note,text width=2.7cm,align=center] at ({x0 + 1.35:.2f},-0.2) {{{why}}};\n")
f.save(__file__)
