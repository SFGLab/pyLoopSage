"""Writes binding_modes.tex: how loops become anchor profiles, and what contrastive binding adds (toy numbers).

Run: python3 binding_modes.py
"""
from pathlib import Path

import numpy as np

N = 40
LOOPS = [(3, 11), (3, 17), (11, 17), (24, 36), (24, 31)]          # toy loops (left bead, right bead); beads 17-24 are loop-free
L, R = np.zeros(N), np.zeros(N)
for m, n in LOOPS:
    L[m] += 1; R[n] += 1
d = np.zeros(N + 1)
for m, n in LOOPS:
    d[m] += 1; d[n] -= 1
cover = np.cumsum(d)[:N]                                           # loops spanning each bead
chat = np.where(cover < 0.05 * cover.mean(), -1.0, np.clip(cover / cover.mean() - 1, 0, 1))   # -1 loop-free, 0 up to the average, +1 at twice the average
Lm, Rm = L / L.mean(), R / R.mean()                                # anchor profiles scaled by their mean (both modes)
lbar = np.mean([n - m for m, n in LOOPS])
cands = [((3, 17), "over covered beads"), ((17, 24), "over loop-free beads"), ((24, 36), "over covered beads")]
X = lambda i, x0, w: x0 + w * i / (N - 1)
W0, WW = 1.2, 11.6

tex = r"""\input{preamble.tex}
\tikzset{ln/.style={line width=0.7pt,ink!45},arc/.style={line width=1.1pt,cohesin!85},cand/.style={line width=2pt,A}}
\definecolor{L}{HTML}{6A3DE8}\definecolor{R}{HTML}{1A1A1F}\definecolor{A}{HTML}{D1303F}\definecolor{B}{HTML}{5B4FD6}
\begin{document}
\begin{tikzpicture}
  \panelbg{0}{-0.2}{13.4}{5.3}\panelbg{0}{-5.4}{13.4}{-0.5}\panelbg{0}{-11.0}{13.4}{-5.7}
  \badge{0.45}{4.85}{a}\node[hd] at (0.85,4.85) {Both modes: loops become anchor profiles $L$ and $R$};
  \badge{0.45}{-1.0}{b}\node[hd] at (0.85,-1.0) {Contrastive mode adds the coverage of the input loops};
  \badge{0.45}{-6.2}{c}\node[hd] at (0.85,-6.2) {What a simulated loop feels};
"""
# ---- a: arcs, L and R
y0 = 2.65
tex += f"  \\draw[ln] ({X(0, W0, WW):.2f},{y0})--({X(N - 1, W0, WW):.2f},{y0});\n"
for m, n in LOOPS:
    xm, xn, h = X(m, W0, WW), X(n, W0, WW), 0.075 * (n - m) * 4 / 3
    tex += f"  \\draw[arc] ({xm:.2f},{y0}) .. controls ({xm:.2f},{y0 + h:.2f}) and ({xn:.2f},{y0 + h:.2f}) .. ({xn:.2f},{y0});\n"
for v, base, col, name in ((L, 1.45, "L", "$L$"), (R, 0.3, "R", "$R$")):
    tex += f"  \\draw[ln] ({X(0, W0, WW):.2f},{base})--({X(N - 1, W0, WW):.2f},{base});\\node[lab,anchor=east] at (1.0,{base + 0.3}) {{{name}}};\n"
    for i in np.flatnonzero(v):
        tex += f"  \\fill[{col}] ({X(i, W0, WW) - 0.06:.2f},{base}) rectangle ({X(i, W0, WW) + 0.06:.2f},{base + 0.35 * v[i]:.2f});\n"
# ---- b: coverage
yb = -3.25
tex += f"  \\draw[ln] ({X(0, W0, WW):.2f},{yb})--({X(N - 1, W0, WW):.2f},{yb});\n"
tex += f"  \\node[note,anchor=east] at (1.0,{yb + 0.9}) {{$+1$}};\\node[note,anchor=east] at (1.0,{yb - 0.9}) {{$-1$}};\\node[note,anchor=east] at (1.0,{yb}) {{0}};\n"
hw = WW / (N - 1) / 2
for v in (-1, 1):
    tex += f"  \\draw[ink!20,dashed,line width=0.4pt] ({X(0, W0, WW) - hw:.2f},{yb + 0.9 * v})--({X(N - 1, W0, WW) + hw:.2f},{yb + 0.9 * v});\n"
for i in range(N):
    h = 0.9 * chat[i] if chat[i] != 0 else 0.03
    tex += f"  \\fill[{'L!75,draw=L!75' if chat[i] >= 0 else 'ink!30,draw=ink!30'},line width=0.3pt] ({X(i, W0, WW) - hw:.2f},{yb}) rectangle ({X(i, W0, WW) + hw:.2f},{yb + h:.2f});\n"
tex += f"  \\node[note,text width=11.4cm,anchor=north west] at (1.2,-4.3) {{for each bead, count the input loops that span it. $-1$ (grey): no loop spans it. $0$: covered, up to the average count. Up to $+1$ (purple): twice the average or more.}};\n"
# ---- c: three candidate loops
for k, ((m, n), name) in enumerate(cands):
    x0 = 0.9 + 4.2 * k
    xs = lambda i: x0 + 3.6 * i / (N - 1)
    yc = -8.55
    tex += f"  \\fill[L!8] ({xs(m):.2f},{yc - 0.45}) rectangle ({xs(n):.2f},{yc + 0.45});\n"       # span of the candidate loop
    tex += f"  \\draw[ln] ({xs(0):.2f},{yc})--({xs(N - 1):.2f},{yc});\n"
    for i in range(N):
        h = 0.4 * chat[i] if chat[i] != 0 else 0.015
        tex += f"  \\fill[{'L!75,draw=L!75' if chat[i] >= 0 else 'ink!30,draw=ink!30'},line width=0.3pt] ({xs(i) - 0.045:.2f},{yc}) rectangle ({xs(i) + 0.045:.2f},{yc + h:.2f});\n"
    hh = 0.75 + 0.045 * (n - m)                                                           # arc height above the fibre
    tex += f"  \\draw[cand] ({xs(m):.2f},{yc}) .. controls ({xs(m):.2f},{yc + hh:.2f}) and ({xs(n):.2f},{yc + hh:.2f}) .. ({xs(n):.2f},{yc});\n"
    for i in (m, n):                                                                       # LEF legs sit on the fibre
        tex += f"  \\filldraw[fill=white,draw=A,line width=1.2pt] ({xs(i):.2f},{yc}) circle (0.09);\n"
    anchors = Lm[m] + Rm[n]
    inner = chat[m:n].sum() / lbar
    tex += f"  \\node[lab] at ({x0 + 1.8:.2f},-6.85) {{\\bfseries {name}}};\n"
    tex += f"  \\node[note,align=center] at ({x0 + 1.8:.2f},-9.35) {{anchors (both modes): ${anchors:+.1f}$\\\\interior (contrastive): ${inner:+.1f}$}};\n"
tex += f"  \\node[note,text width=11.8cm,anchor=north west] at (0.9,-10.0) {{Shaded: the beads inside the candidate loop. Anchor values are the same in both modes; only the contrastive mode adds the interior term.}};\n"
tex += r"""\end{tikzpicture}
\end{document}
"""
Path(__file__).with_suffix(".tex").write_text(tex)
print([(c, Lm[c[0][0]] + Rm[c[0][1]], chat[c[0][0]:c[0][1]].sum() / lbar) for c in cands])
