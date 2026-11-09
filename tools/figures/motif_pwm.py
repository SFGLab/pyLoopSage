"""Writes motif_pwm.tex: CTCF logo, scoring of one window, and the exact random-score distribution with the p-value cutoff.

All numbers come from the bundled JASPAR MA0139.1 matrix and src/motif_finder (run: python3 motif_pwm.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from loopsage.motif_finder.fasta import encode            # noqa: E402
from loopsage.motif_finder.pssm import load_pfm           # noqa: E402

P = load_pfm()                                   # default matrix, pseudocount 0.8
L, M = P.length, P.M
prob = 0.25 * 2 ** M                             # probabilities per base and position
info = 2 + (prob * np.log2(prob)).sum(axis=0)    # information content of every column (bits)
CUT = P.cutoff(1e-5)
COL = {"A": "ok", "C": "cohesin", "G": "ink", "T": "orange"}
X = lambda i: 1.55 + 0.6 * i

# example window: the consensus with two changes (second-best bases at weakly informative positions) that still scores above the cutoff
cons = P.consensus()
second = ["ACGT"[np.argsort(M[:, i])[-2]] for i in range(L)]
best = None
for i in range(L):
    for j in range(i + 1, L):
        w = list(cons); w[i], w[j] = second[i], second[j]
        sc_ = M[encode("".join(w)), np.arange(L)]
        if CUT + 1 <= sc_.sum() <= CUT + 6 and (best is None or sc_.sum() < best[0]):
            best = (sc_.sum(), "".join(w), sc_)
S, win, sc = best

# random-score distribution (exact, same recursion as Pssm.cutoff)
res = 100
q = np.round(M * res).astype(int)
dist = np.ones(1)
for i in range(L):
    lo = q[:, i].min()
    new = np.zeros(len(dist) + q[:, i].max() - lo)
    for b in range(4):
        new[q[b, i] - lo:q[b, i] - lo + len(dist)] += 0.25 * dist
    dist = new
xs = (q.min(axis=0).sum() + np.arange(len(dist))) / res
edges = np.arange(-70, 30, 2.0)
binp = np.array([dist[(xs >= a) & (xs < a + 2)].sum() for a in edges])

T = []
T.append(r"""\input{preamble.tex}
\definecolor{orange}{HTML}{E8A33D}
\tikzset{ln/.style={line width=0.7pt,ink!45}}
\begin{document}
\begin{tikzpicture}
  \panelbg{0}{-3.3}{13.4}{0.9}\panelbg{0}{-7.5}{13.4}{-3.6}\panelbg{0}{-14.5}{13.4}{-7.8}
  \badge{0.45}{0.5}{a}\node[hd] at (0.85,0.5) {The motif as a position weight matrix: CTCF (JASPAR MA0139.1, 19 bp)};
  \badge{0.45}{-4.05}{b}\node[hd] at (0.85,-4.05) {Scoring one window of DNA};
  \badge{0.45}{-8.4}{c}\node[hd] at (0.85,-8.4) {How rare is a score? Random DNA};
""")
# a: logo (letter heights = probability x information of the column)
T.append(r"  \draw[ln] (1.0,-2.5)--(13.0,-2.5);")
for t in (0, 1, 2):
    T.append(rf"  \node[note,anchor=east] at (0.95,{-2.5 + t * 1.0:.2f}) {{{t if t < 2 else '2 bits'}}};\draw[ln] (1.0,{-2.5 + t * 1.0:.2f})--(1.1,{-2.5 + t * 1.0:.2f});")
for i in range(L):
    base = -2.5
    for b in np.argsort(prob[:, i] * info[i]):                       # smallest on the bottom
        h = prob[b, i] * info[i] * 1.0
        if h < 0.05:
            continue
        T.append(rf"  \node[anchor=south,inner sep=0,yscale={h / 0.5:.3f},xscale=1.05,text={COL['ACGT'[b]]},font=\bfseries\Large] at ({X(i):.2f},{base:.3f}) {{{'ACGT'[b]}}};")
        base += h
    T.append(rf"  \node[note] at ({X(i):.2f},-2.8) {{{i + 1}}};")
T.append(r"  \node[note,anchor=east] at (13.2,0.0) {tall letter = base that is almost always there};")
# b: one window
y0 = -5.0
T.append(rf"  \node[note,anchor=east] at (1.0,{y0}) {{DNA}};\node[note,anchor=east] at (1.0,{y0 - 0.55}) {{score}};")
for i in range(L):
    same = win[i] == cons[i]
    T.append(rf"  \node[font=\bfseries\large,text={'ink' if same else 'fwd'}] at ({X(i):.2f},{y0}) {{{win[i]}}};")
    T.append(rf"  \node[font=\scriptsize,text={'ink!70' if sc[i] >= 0 else 'fwd'}] at ({X(i):.2f},{y0 - 0.55}) {{{sc[i]:+.1f}}};")
T.append(rf"  \node[lab,anchor=west] at (1.2,{y0 - 1.25}) {{$S=\sum_i \log_2\frac{{P(\text{{base}}_i)}}{{0.25}}={S:.1f}$}};")
T.append(rf"  \node[lab,anchor=west,text=ok] at (8.4,{y0 - 1.25}) {{$\ge t={CUT:.1f}$: a hit}};")
T.append(rf"  \node[note,anchor=west] at (1.2,{y0 - 1.95}) {{red: differs from the best base; the other positions compensate, so the window is still a hit}};")
# c: log-scale histogram
x0, y0c, W, H = 2.0, -13.1, 10.4, 3.0
lg = np.log10(np.maximum(binp, 1e-12))
lo_, hi_ = -9.0, 0.0
T.append(rf"  \draw[ln] ({x0},{y0c})--({x0 + W},{y0c});\draw[ln] ({x0},{y0c})--({x0},{y0c + H});")
for tick in (0, -3, -6, -9):
    yy = y0c + (tick - lo_) / (hi_ - lo_) * H
    T.append(rf"  \draw[ln] ({x0 - 0.08},{yy:.2f})--({x0},{yy:.2f});\node[note,anchor=east] at ({x0 - 0.1},{yy:.2f}) {{$10^{{{tick}}}$}};")
for a, v in zip(edges, lg):
    h = (max(v, lo_) - lo_) / (hi_ - lo_) * H
    if h <= 0.01:
        continue
    xa, xb = x0 + (a + 70) / 100 * W, x0 + (a + 72) / 100 * W
    col = "cohesin" if a + 2 > CUT else "cohesin!30"
    T.append(rf"  \fill[{col}] ({xa:.3f},{y0c}) rectangle ({xb:.3f},{y0c + h:.3f});")
xt = x0 + (CUT + 70) / 100 * W
T.append(rf"  \draw[ink,line width=1pt,dash pattern=on 3pt off 2pt] ({xt:.3f},{y0c})--({xt:.3f},{y0c + H + 0.15});")
T.append(rf"  \node[lab,anchor=south] at ({xt:.3f},{y0c + H + 0.15}) {{$t$, where $P(S\ge t)=10^{{-5}}$}};")
for s in (-60, -40, -20, 0, 20):
    xx = x0 + (s + 70) / 100 * W
    T.append(rf"  \draw[ln] ({xx:.2f},{y0c})--({xx:.2f},{y0c - 0.08});\node[note,anchor=north] at ({xx:.2f},{y0c - 0.1}) {{{s}}};")
T.append(rf"  \node[note,anchor=north] at ({x0 + W / 2},{y0c - 0.55}) {{score $S$ of a random window}};\node[note,anchor=east,rotate=90] at ({x0 - 1.25},{y0c + H / 2}) {{probability}};")
T.append(r"\end{tikzpicture}\end{document}")
Path(__file__).with_name("motif_pwm.tex").write_text("\n".join(T) + "\n")
print("window", win, "S=%.2f" % S, "cutoff %.2f" % CUT)
