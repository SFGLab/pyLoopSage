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
# ---- a: rules -> ensemble -> map, all centred on one line (y = YC)
YC, YH = 6.55, 8.7
for x, t in ((2.3, "Rules"), (6.7, "Many random loop states"), (11.1, "Average contact map")):
    f.text(x, YH, "{\\bfseries " + t + "}", "lab")
f.add(f"  \\node[box,minimum width=3.5cm,minimum height=2.7cm,text width=2.9cm,align=left] at (2.3,{YC}) {{\\footnotesize a ring loads on the fibre\\\\[4pt]\\footnotesize it extrudes a loop\\\\[4pt]\\footnotesize CTCF can stop it\\\\[4pt]\\footnotesize rings never cross}};\n")
for k in range(3):
    y = YC + 0.9 - 0.9 * k
    f.add(f"  \\draw[ln] (5.3,{y})--(8.1,{y});\n")
    loops = [[(0.2, 1.4)], [(0.6, 2.2), (1.0, 1.7)], [(1.4, 2.5), (0.2, 0.9)]][k]
    for a, b in loops:
        h = 0.2 * (b - a)
        f.add(f"  \\draw[cohesin,thin] ({5.3 + a},{y}) .. controls ({5.3 + a},{y + h * 1.6:.2f}) and ({5.3 + b},{y + h * 1.6:.2f}) .. ({5.3 + b},{y});\n")
f.add(f"  \\draw[mv] (4.25,{YC})--(5.0,{YC});\\node[note] at (4.62,{YC - 0.35}) {{sample}};\n  \\draw[mv] (8.4,{YC})--(9.15,{YC});\\node[note] at (8.78,{YC - 0.35}) {{average}};\n")
mini(f, 9.5, YC - 1.6, 3.2, toy_map())
# ---- b: four cards, equal width, the last one marked as needing an extra ingredient
cards = [("tad", "TAD", "loops grow between two CTCF sites"), ("dot", "Loop dot", "a ring stalls between two convergent CTCF"),
         ("stripe", "Stripe", "one leg rests at CTCF, the other keeps extruding"), ("comp", "A/B compartments", "need an attraction between like chromatin")]
for k, (kind, name, why) in enumerate(cards):
    x0 = 0.35 + 3.2 * k
    edge = "B!55,dashed" if kind == "comp" else "panelline"
    f.add(f"  \\draw[rounded corners=6pt,fill=white,draw={edge},line width=0.9pt] ({x0},-0.95) rectangle ({x0 + 3.0},3.45);\n")
    f.text(x0 + 1.5, 3.08, "{\\bfseries " + name + "}", "lab")
    mini(f, x0 + 0.6, 0.85, 1.8, pattern(kind), "cohesin" if kind != "comp" else "B")
    f.add(f"  \\node[note,text width=2.6cm,align=center] at ({x0 + 1.5:.2f},0.12) {{{why}}};\n")
f.add("  \\node[font=\\scriptsize\\bfseries,text=B!80,fill=white,inner sep=1.5pt] at (%.2f,-0.95) {extra ingredient};\n" % (0.35 + 3.2 * 3 + 1.5))
f.save(__file__)
