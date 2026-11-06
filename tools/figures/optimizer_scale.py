"""Writes optimizer_scale.tex: how the loss weights follow the region size, what the region contains, and the two optimizer stages.

The weights come from src/optimizer/objective.py (run: python3 optimizer_scale.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from optimizer.objective import loss_weights   # noqa: E402

X = lambda mb: 1.5 + 4.6 * np.log10(mb)        # log axis: 1 Mb -> 1.5, 250 Mb -> 12.5 (cm)
Y = lambda w: 1.0 + 1.6 * w                    # weight 0 -> 1.0, 2 -> 4.2
S = np.logspace(0, np.log10(250), 60)
pc1 = " ".join(f"({X(s):.3f},{Y(loss_weights(s * 1e6)[0]):.3f})" for s in S)
ins = " ".join(f"({X(s):.3f},{Y(loss_weights(s * 1e6)[2]):.3f})" for s in S)
w17 = loss_weights(17e6)
cols = [(5, "$\\approx$5 Mb", "a few TADs", "1 A/B block"), (15, "$\\approx$15 Mb", "more, smaller TADs", "3--4 A/B blocks"), (60, "$\\ge$30 Mb", "tiny TADs", "many A/B blocks")]
wt = [loss_weights(m * 1e6) for m, *_ in cols]

tex = r"""\input{preamble.tex}
\tikzset{ln/.style={line width=0.7pt,ink!45},bead/.style={circle,inner sep=0pt,minimum size=0.2cm,draw=white,line width=0.3pt},
  bx/.style={rounded corners=6pt,draw=cohesin!70,fill=cohesin!8,line width=0.9pt,align=center,font=\small,text=ink},
  ar/.style={line width=1.4pt,ink!70,-{Stealth[length=6pt,width=7pt]}}}
\definecolor{A}{HTML}{D1303F}\definecolor{B}{HTML}{5B4FD6}

\begin{document}
\begin{tikzpicture}
  \panelbg{0}{-0.2}{13.4}{5.7}\panelbg{0}{-7.4}{13.4}{-0.6}\panelbg{0}{-11.8}{13.4}{-7.8}
  \badge{0.45}{5.25}{a}\node[hd] at (0.85,5.25) {Loss weights follow the size of the modelled region};
  \badge{0.45}{-1.2}{b}\node[hd] at (0.85,-1.2) {What the region contains, and what the loss can see in it};
  \badge{0.45}{-8.4}{c}\node[hd] at (0.85,-8.4) {With the Potts model on, two searches in a row};
  %% a: weights on a log axis
  \fill[cohesin!6] (%(x1).3f,1.0) rectangle (%(x5).3f,4.95);\fill[cohesin!12] (%(x5).3f,1.0) rectangle (%(x30).3f,4.95);\fill[cohesin!20] (%(x30).3f,1.0) rectangle (%(x250).3f,4.95);
  \draw[ln] (1.5,1.0)--(12.5,1.0);\draw[ln] (1.5,1.0)--(1.5,4.2);
  \foreach \w in {0,1,2}{\draw[ln] (1.4,{1.0+1.6*\w})--(1.5,{1.0+1.6*\w});\node[note,anchor=east] at (1.35,{1.0+1.6*\w}) {\w};}
  \node[note,rotate=90] at (0.7,2.6) {weight in the loss};
  \foreach \m in {1,5,10,30,100,250}{\draw[ln] ({1.5+4.6*log10(\m)},1.0)--({1.5+4.6*log10(\m)},0.9);\node[note] at ({1.5+4.6*log10(\m)},0.62) {\m};}
  \node[note] at (7.0,0.2) {region length (Mb, log scale)};
  \draw[ln,dashed] (%(x5).3f,1.0)--(%(x5).3f,4.95);\draw[ln,dashed] (%(x30).3f,1.0)--(%(x30).3f,4.95);
  \node[lab] at (%(z1).3f,4.7) {a few TADs};\node[lab] at (%(z2).3f,4.7) {a few A/B blocks};\node[lab] at (%(z3).3f,4.7) {many A/B blocks};
  \draw[line width=2.2pt,cohesin] plot[smooth] coordinates {%(pc1)s};
  \draw[line width=2.2pt,ink] plot[smooth] coordinates {%(ins)s};
  \node[lab,text=cohesin,anchor=west] at (1.75,1.45) {PC1 term};\node[lab,anchor=west] at (1.75,3.85) {insulation term};
  \fill[cohesin] (%(x17).3f,%(y17a).3f) circle (0.09);\fill[ink] (%(x17).3f,%(y17b).3f) circle (0.09);
  \node[note,fill=white,inner sep=1.5pt,align=center] at (5.9,2.6) {17 Mb:\\{%(w17a).1f} and {%(w17b).1f}};
""" % dict(x1=X(1), x5=X(5), x30=X(30), x250=X(250), z1=(X(1) + X(5)) / 2, z2=(X(5) + X(30)) / 2, z3=(X(30) + X(250)) / 2, pc1=pc1, ins=ins, x17=X(17),
           y17a=Y(w17[0]), y17b=Y(w17[2]), w17a=w17[0], w17b=w17[2])

# ---- b: three scales, each with a domain/compartment strip and a 3D sketch
strips = {0: "A", 1: "AB", 2: "ABAB"}
for i, (m, size, tads, blocks) in enumerate(cols):
    x0 = 0.6 + 4.2 * i
    y0 = -2.75
    nb = [1, 3, 8][i]
    wb = 3.6 / nb
    nt = [3, 7, 14][i]
    tex += f"  \\node[lab] at ({x0 + 1.8},-1.75) {{\\bfseries {size}}};\n"
    for k in range(nt):                                  # domain triangles
        a, b = x0 + 3.6 * k / nt, x0 + 3.6 * (k + 1) / nt
        tex += f"  \\draw[ink!70,line width=0.6pt,fill=white] ({a + 0.04:.2f},{y0}) -- ({(a + b) / 2:.2f},{y0 + 0.55 * (b - a) / 1.2:.2f}) -- ({b - 0.04:.2f},{y0}) -- cycle;\n"
    for k in range(nb):                                  # A/B blocks
        c = "A" if k % 2 == 0 else "B"
        tex += f"  \\fill[{c}] ({x0 + wb * k:.2f},{y0 - 0.35}) rectangle ({x0 + wb * (k + 1):.2f},{y0 - 0.12});\n"
    tex += f"  \\node[note,align=center] at ({x0 + 1.8},{y0 - 0.85}) {{{tads}\\\\{blocks}}};\n"
y3 = -4.75
# 3D sketches: a smooth fibre with equally spaced beads, coloured by compartment block (a: three TAD globules, b: open chain with A, B, A blocks, c: one loose A aggregate, one dense B aggregate)
from scipy.interpolate import splprep, splev


def scribble(cx, cy, r, n, seed):
    """Control points of a fibre wandering inside a disc of radius r (a compact globule)."""
    rng = np.random.default_rng(seed)
    out, p = [], np.array([cx + r, cy])
    for _ in range(n):
        p = p + 0.5 * (np.array([cx, cy]) - p) + rng.normal(0, r * 0.6, 2)
        out.append(p.copy())
    return out


def chain(x0, y0, parts, d=0.2):
    """parts = [(control points, colour)]: one smooth fibre through all of them, a bead every d cm along it, bead colour = colour of the nearest part."""
    pts = [q for c, _ in parts for q in c]
    tag = [col for c, col in parts for _ in c]
    tck, u = splprep(np.array(pts).T, k=3, s=0.02)
    g = np.linspace(0, 1, 2000)
    xy = np.array(splev(g, tck)).T
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    path = " ".join(f"({x0 + a:.3f},{y0 + b:.3f})" for a, b in xy[::8])
    out = f"  \\draw[fiber,line width=1.5pt,ink!75,line join=round] plot[smooth,tension=0.5] coordinates {{{path}}};\n"
    for a in np.arange(d / 2, arc[-1], d):
        k = np.searchsorted(arc, a)
        c = tag[int(np.argmin(np.abs(u - g[k])))]
        out += f"  \\node[bead,fill={c},minimum size=0.17cm] at ({x0 + xy[k, 0]:.3f},{y0 + xy[k, 1]:.3f}) {{}};\n"
    return out


G, RA, RB = "ink!45", "A!85!white", "B!85!white"
gl = [scribble(dx, 0, 0.33, 9, 20 + k) for k, dx in enumerate((-1.35, 0.0, 1.35))]
c1 = [(sum(gl, []) and gl[k], G) for k in range(3)]
c2 = [([(-1.6 + 3.2 * k / 14, 0.4 * np.sin(k * 0.9)) for k in range(a, b)], c) for a, b, c in ((0, 5, RA), (5, 10, RB), (10, 15, RA))]
la, lb = (0, 0.3), (0, -0.3)
c3 = [(scribble(-0.85, 0.0, 0.62, 9, 31), RA), (scribble(0.8, 0.0, 0.27, 14, 32), RB)]   # A: loose aggregate, B: small dense one, joined directly
tex += chain(2.4, y3, c1) + chain(6.6, y3, c2) + chain(10.8, y3, c3)
for i, w in enumerate(wt):
    x0 = 0.6 + 4.2 * i
    tex += f"  \\node[lab,align=center] at ({x0 + 1.8},-6.0) {{PC1 weight {w[0]:.2f}\\\\insulation weight {w[2]:.2f}}};\n"
tex += "  \\node[note] at (6.7,-6.95) {chain sketches: grey = TAD globules, red = A beads, blue = B beads};\n"
# ---- c: two stages
tex += r"""  \node[bx,minimum width=5.6cm,minimum height=2.6cm] (s1) at (3.3,-10.0) {\bfseries 1\; Loop extrusion\\[2pt]{\footnotesize Potts model off}\\[3pt]fits $K,\ f,\ b,\ T$\\[3pt]{\footnotesize insulation, decay, loop size}};
  \node[bx,minimum width=5.6cm,minimum height=2.6cm,draw=ink!70,fill=white] (s2) at (10.1,-10.0) {\bfseries 2\; Potts model\\[2pt]{\footnotesize loop-extrusion values fixed}\\[3pt]fits {\small\texttt{epi\_coeff}, \texttt{p\_epigenetic}}\\[3pt]{\footnotesize full loss: PC1 of $\langle s\rangle_t$ balanced with the TAD terms}};
  \draw[ar] (s1)--(s2) node[midway,above,font=\footnotesize,text=ink!70] {};
\end{tikzpicture}
\end{document}
"""
Path(__file__).with_suffix(".tex").write_text(tex)
