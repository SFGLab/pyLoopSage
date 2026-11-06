"""Writes mech_equilibrium.tex: detailed balance, a driven cycle (no detailed balance), and an effective energy from a stationary distribution.

Run: python3 mech_equilibrium.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "Detailed balance: no net flow between any two states", 0, 9.0, 13.4, 13.4)
f.panel("b", "A driven system: a net current around a cycle", 0, 3.3, 13.4, 8.8)
f.panel("c", "The LoopSage approximation: an effective energy", 0, -1.7, 13.4, 3.1)
# ---- a: two states
f.add("  \\fill[cohesin!25] (2.2,10.6) circle (0.75);\\draw[cohesin,line width=1pt] (2.2,10.6) circle (0.75);\\node[lab] at (2.2,10.6) {$i$};\n")
f.add("  \\fill[cohesin!12] (6.0,10.6) circle (0.5);\\draw[cohesin,line width=1pt] (6.0,10.6) circle (0.5);\\node[lab] at (6.0,10.6) {$j$};\n")
f.add("  \\draw[mv,line width=2.2pt] (3.05,10.9) .. controls (4.0,11.4) and (4.8,11.4) .. (5.4,11.05);\n  \\draw[mv,line width=2.2pt] (5.4,10.15) .. controls (4.8,9.8) and (4.0,9.8) .. (3.05,10.3);\n")
f.add("  \\node[note] at (4.2,11.75) {flow $i\\to j$: $\\pi_i P_{ij}$};\\node[note] at (4.2,9.4) {flow $j\\to i$: $\\pi_j P_{ji}$};\n")
f.add("  \\node[note,text width=5.3cm,anchor=north west] at (7.7,12.4) {At equilibrium the two flows are equal for each pair:\\\\[3pt]$\\pi_i P_{ij}=\\pi_j P_{ji}$\\\\[3pt]Nothing circulates, and a film of the system looks the same played backwards. The disc size shows the probability $\\pi$ of the state.};\n")
# ---- b: cycle with current
cx, cy, r = 3.0, 5.2, 1.3
pts = [(cx + r * np.cos(np.pi / 2 + 2 * np.pi * k / 3), cy + r * np.sin(np.pi / 2 + 2 * np.pi * k / 3)) for k in range(3)]
names = ["load", "extrude", "release"]
for k, (x, y) in enumerate(pts):
    f.add(f"  \\fill[cohesin!18] ({x:.2f},{y:.2f}) circle (0.42);\\draw[cohesin,line width=1pt] ({x:.2f},{y:.2f}) circle (0.42);\n")
for k in range(3):
    a, b = pts[k], pts[(k + 1) % 3]
    ang_a = np.arctan2(b[1] - a[1], b[0] - a[0])
    s0 = (a[0] + 0.5 * np.cos(ang_a), a[1] + 0.5 * np.sin(ang_a)); s1 = (b[0] - 0.5 * np.cos(ang_a), b[1] - 0.5 * np.sin(ang_a))
    f.add(f"  \\draw[mv,line width=2.4pt,cohesin] ({s0[0]:.2f},{s0[1]:.2f})--({s1[0]:.2f},{s1[1]:.2f});\n")
    f.add(f"  \\draw[mv,line width=0.6pt,ink!40] ({s1[0] + 0.12 * np.sin(ang_a):.2f},{s1[1] - 0.12 * np.cos(ang_a):.2f})--({s0[0] + 0.12 * np.sin(ang_a):.2f},{s0[1] - 0.12 * np.cos(ang_a):.2f});\n")
f.add(f"  \\node[note] at ({pts[0][0] + 0.9:.2f},{pts[0][1] + 0.0:.2f}) {{load}};\\node[note] at ({pts[1][0] - 0.95:.2f},{pts[1][1] - 0.2:.2f}) {{extrude}};\\node[note] at ({pts[2][0] + 0.95:.2f},{pts[2][1] - 0.2:.2f}) {{release}};\n")
f.add("  \\node[note,text width=7.2cm,anchor=north west] at (5.6,7.5) {A cohesin ring loads, extrudes (it burns \\textbf{ATP} to move) and is released. The forward arrows (thick) are much stronger than the backward ones (thin): there is a steady \\textbf{current} around the cycle, so $\\pi_iP_{ij}\\neq\\pi_jP_{ji}$. The system is \\textbf{out of equilibrium}; it is in a \\emph{steady state}, but not in detailed balance.};\n")
# ---- c: effective energy
l = np.linspace(0.05, 6, 80)
P = np.exp(-l / 1.6) * l ** 0.3; P /= P.max()
Ee = -np.log(P + 1e-9); Ee -= Ee.min()
axP = Axes(f, 1.5, -0.7, 3.0, 1.9, (0, 6), (0, 1.1), "loop length", "stationary $\\pi(\\ell)$")
axP.line(l, P, "cur", 2)
axE = Axes(f, 6.9, -0.7, 3.0, 1.9, (0, 6), (0, 5.2), "loop length", "effective $E(\\ell)$")
axE.line(l, np.minimum(Ee, 5.1), "cur", 2)
f.add("  \\draw[mv] (4.9,0.3)--(5.8,0.3);\\node[note] at (5.35,0.7) {$-T\\ln\\pi$};\n")
f.add("  \\node[note,text width=2.5cm,anchor=north west] at (10.5,1.5) {Measure or predict the stationary distribution, then write it as $\\pi\\propto e^{-E/T}$.};\n")
f.save(__file__)
