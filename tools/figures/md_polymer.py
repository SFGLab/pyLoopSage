"""Writes md_polymer.tex: coarse-graining chromatin into beads, the bead-spring chain, and how its size scales with length.

Run: python3 md_polymer.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "From a fibre to beads: coarse-graining", 0, 8.4, 13.4, 12.6)
f.panel("b", "A bead-spring chain", 0, 3.6, 13.4, 8.2)
f.panel("c", "Size of a polymer chain with $N$ beads", 0, -1.6, 13.4, 3.4)
# ---- a: wiggly fibre and its beads
t = np.linspace(0, 1, 200)
x = 1.2 + 6.4 * t; y = 11.0 + 0.35 * np.sin(14 * t) + 0.2 * np.sin(31 * t)
f.add("  \\draw[line width=1.4pt,ink!70] " + " -- ".join(f"({a:.2f},{b:.2f})" for a, b in zip(x, y)) + ";\n")
f.add("  \\node[note,anchor=west] at (8.4,11.4) {real fibre: DNA on nucleosomes};\n")
for i in range(8):
    xb = 1.6 + 0.8 * i
    f.add(f"  \\fill[cohesin!{40 + 8 * (i % 3)}] ({xb:.2f},9.6) circle (0.3);\\draw[cohesin,line width=0.8pt] ({xb:.2f},9.6) circle (0.3);\n")
    if i: f.add(f"  \\draw[ink!60,line width=1pt] ({xb - 0.5:.2f},9.6)--({xb - 0.3:.2f},9.6);\n")
f.add("  \\node[note,anchor=west] at (8.4,9.6) {the model: one bead = a few kb};\n")
f.add("  \\draw[mv] (4.6,10.5)--(4.6,10.0);\n")
f.add("  \\node[note,text width=11.8cm,anchor=north west] at (0.9,9.05) {Details below the bead size are not simulated. Only the position of each bead is followed.};\n")
# ---- b: chain with bond, angle, excluded volume
pts = [(1.3, 5.2), (2.5, 6.0), (3.8, 5.5), (5.0, 6.4), (6.0, 5.4), (5.4, 4.4), (4.2, 4.6)]
for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
    f.add(f"  \\draw[ink!55,line width=1.6pt] ({x0},{y0})--({x1},{y1});\n")
for k, (x0, y0) in enumerate(pts):
    f.add(f"  \\fill[cohesin!55] ({x0},{y0}) circle (0.3);\\draw[cohesin,line width=0.8pt] ({x0},{y0}) circle (0.3);\n")
f.add("  \\node[note] at (1.9,4.5) {bond};\\draw[leader] (1.9,4.75)--(1.9,5.6);\n")
f.add("  \\draw[cohesin,line width=1.3pt] ([shift={(3.8,5.5)}]37:0.75) arc[start angle=37,end angle=159,radius=0.75];\n")
f.add("  \\node[lab,text=cohesin] at (3.75,6.55) {$\\theta$};\n")
f.add("  \\node[note,anchor=west] at (1.9,7.1) {angle $\\theta$ between two bonds};\n")
f.add("  \\draw[ink!60,dashed,line width=0.9pt] (6.0,5.4)--(4.2,4.6);\\node[note,anchor=west] at (1.9,3.95) {dashed: beads keep apart};\n")
f.add("  \\node[note,text width=5.8cm,anchor=north west] at (7.4,7.7) {Each bead has a position $\\mathbf r_i$. The chain is a set of \\textbf{springs} between neighbours (bonds) plus a few rules: stiffness (angle), and no two beads in the same place (\\textbf{excluded volume}).};\n")
f.add("  \\node[note,text width=5.8cm,anchor=north west] at (7.4,5.6) {The shape of the chain is not drawn by hand: it follows from the forces and from thermal motion.};\n")
# ---- c: scaling
n = np.array([10, 1000])
ax = Axes(f, 1.6, -0.7, 4.2, 3.0, (1, 3), (0, 2), "$\\log_{10} N$", "$\\log_{10} R$", log=True, xticks=(1, 2, 3), yticks=())
for nu, name, col in ((0.588, "swollen coil", "cohesin"), (0.5, "ideal chain", "cohesin!55"), (1 / 3, "compact globule", "ink!70")):
    ax.line([1, 3], [0.2, 0.2 + 2 * nu * 0.9], f"line width=1.8pt,{col}", 1)
f.add("  \\node[note,anchor=west,text=cohesin] at (6.0,1.9) {$R\\sim N^{0.59}$: swollen coil};\n")
f.add("  \\node[note,anchor=west,text=cohesin!70] at (6.0,1.2) {$R\\sim N^{1/2}$: ideal chain};\n")
f.add("  \\node[note,anchor=west,text=ink!75] at (6.0,0.5) {$R\\sim N^{1/3}$: compact globule};\n")
f.add("  \\node[note,text width=3.4cm,anchor=north west] at (9.8,2.6) {$R$ is the size of the chain; $\\nu$ in $R\\sim N^\\nu$ is its \\textbf{scaling exponent}.};\n")
f.save(__file__)
